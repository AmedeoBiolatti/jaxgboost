import jax
from jax.typing import ArrayLike
from dataclasses import dataclass

from jaxgboost.tree_builders import base
from jaxgboost import losses
from jaxgboost.trees.tree import GHTree


@jax.tree_util.register_dataclass
@dataclass
class Struct:
    tree: GHTree
    rank_start: ArrayLike
    rank_end: ArrayLike
    rank: ArrayLike
    rank2index: ArrayLike
    first_free_id: int
    next_node_to_expand: ArrayLike = 0
    should_continue: ArrayLike = True

    @classmethod
    def init(cls, rank2index, num_nodes: int, num_targets: int):
        n_obs, n_col = rank2index.shape
        rank_start = jax.numpy.zeros((num_nodes,), dtype=jax.numpy.uint32)
        rank_end = jax.numpy.zeros((num_nodes,), dtype=jax.numpy.uint32) + n_obs
        rank = jax.numpy.zeros((num_nodes,), dtype=jax.numpy.uint32) + n_obs
        first_free_id = 0
        tree = GHTree.init(num_nodes, num_targets)
        return cls(
            tree=tree,
            rank_start=rank_start,
            rank_end=rank_end,
            rank=rank,
            rank2index=rank2index,
            first_free_id=first_free_id
        )


def update_rank2index(rank2index, x, col, thr, rank, rank_start, rank_end):
    def update_col(col_to_update, new_rank2index):
        def step(rank, state):
            kt, kf, v = state
            i = rank2index[rank, col_to_update]
            m = x[i, col] <= thr

            vr = rank2index[rank, col_to_update]

            v = jax.lax.cond(
                m,
                lambda: v.at[kt, col_to_update].set(vr),
                lambda: v.at[kf, col_to_update].set(vr)
            )

            kt = kt + (m).astype(jax.numpy.uint32)
            kf = kf + (~m).astype(jax.numpy.uint32)

            state = kt, kf, v
            return state

        nt = rank - rank_start
        kt = rank_start
        kf = rank_start + nt

        new_rank2index = jax.lax.fori_loop(rank_start, rank_end, step, (kt, kf, new_rank2index))[2]
        return new_rank2index

    new_rank2index = rank2index
    new_rank2index = jax.lax.fori_loop(
        jax.numpy.uint32(0),
        jax.numpy.uint32(new_rank2index.shape[1]),
        update_col,
        new_rank2index
    )
    return new_rank2index


class LossGuideTreeBuilder(base.TreeBuilder):
    def __init__(
            self,
            objective: str | losses.Loss = 'l2',
            reg_lambda: float = 1.0,
            reg_alpha: float = 0.0,
            min_split_loss: float = 0.0,
            max_depth: int = 6,
            min_child_weight: float = 0.0,
            num_leaves: int = 65,
            verbosity=0
    ):
        super().__init__(
            objective=objective,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            min_split_loss=min_split_loss,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            num_leaves=num_leaves
        )
        self.num_nodes = 4 * self.num_leaves  # 1 splits, 1 for leaves, 2 for candidate leaves
        self.verbosity = verbosity

    def build_tree(
            self,
            x: ArrayLike,
            y: ArrayLike,
            p: ArrayLike | None = None,
            sample_weight: ArrayLike | None = None,
            *,
            aux_data: dict[str, ArrayLike] | None = None,
            **kwargs,
    ) -> GHTree:
        x, y, sample_weight, p, aux_data, gh = self.init_data(x, y, sample_weight, p, aux_data)
        rank2index = aux_data['rank2index']

        struct = self.init_struct(gh, rank2index)
        struct = self.expand_node(x, gh, struct, node_id=0)

        def update_one_leaf(struct_: Struct) -> Struct:
            if self.verbosity > 0:
                nid = struct_.next_node_to_expand
                jax.debug.print(
                    "splitting node {nid} with gain {gain:.4f} and {n_obs} observations",
                    nid=nid,
                    gain=struct_.tree.gain[nid],
                    n_obs=struct_.rank_end[nid] - struct_.rank_start[nid]
                )
            struct_ = self.from_leaf_to_split(struct_, x, gh, node_id=struct_.next_node_to_expand)

            gain = struct_.tree.gain
            gain += jax.numpy.where(struct_.tree.is_leaf, 0.0, -jax.numpy.inf)
            gain += jax.numpy.where(struct_.tree.depth < self.max_depth, 0.0, -jax.numpy.inf)
            gain += jax.numpy.where(
                struct_.rank_end - struct_.rank_start >= 2 * self.min_child_weight, 0.0, -jax.numpy.inf
            )
            struct_.next_node_to_expand = jax.numpy.argmax(gain)
            next_node_gain = gain[struct_.next_node_to_expand]

            struct_.should_continue &= next_node_gain >= self.min_split_loss
            struct_.should_continue &= next_node_gain > 0.0
            struct_.should_continue &= jax.numpy.sum(struct_.tree.is_leaf) < self.num_leaves
            return struct_

        def condition(struct_: Struct) -> bool:
            return struct_.should_continue

        struct = jax.lax.while_loop(condition, update_one_leaf, struct)
        return struct.tree

    def from_leaf_to_split(self, struct: Struct, x: ArrayLike, gh: ArrayLike, node_id) -> Struct:
        struct.tree.is_leaf = struct.tree.is_leaf.at[node_id].set(False)
        struct.tree.is_split = struct.tree.is_split.at[node_id].set(True)
        col = struct.tree.col[node_id]
        thr = struct.tree.thr[node_id]

        struct.rank2index = update_rank2index(
            struct.rank2index,
            x=x,
            col=col,
            thr=thr,
            rank_start=struct.rank_start[node_id],
            rank_end=struct.rank_end[node_id],
            rank=struct.rank[node_id],
        )

        l_child_id = struct.tree.l_child_id[node_id]
        r_child_id = struct.tree.r_child_id[node_id]

        struct.tree.is_leaf = struct.tree.is_leaf.at[l_child_id].set(True)
        struct = self.expand_node(x, gh, struct, l_child_id)
        struct.tree.is_leaf = struct.tree.is_leaf.at[r_child_id].set(True)
        struct = self.expand_node(x, gh, struct, r_child_id)
        return struct

    def expand_node(self, x: ArrayLike, gh: ArrayLike, struct: Struct, node_id: int) -> Struct:
        score_l, score_r, col, thr, rank, gh_l, gh_r = self.best_split_fori(
            x,
            struct.rank2index,
            gh,
            struct.tree.gh_sum[node_id],
            rank_start=struct.rank_start[node_id],
            rank_end=struct.rank_end[node_id],
            node_id=node_id
        )

        l_child_id = struct.first_free_id
        r_child_id = struct.first_free_id + 1
        struct.first_free_id += 2

        struct.tree.l_child_id = struct.tree.l_child_id.at[node_id].set(l_child_id)
        struct.tree.r_child_id = struct.tree.r_child_id.at[node_id].set(r_child_id)

        # node
        gain = score_l + score_r - struct.tree.score[node_id]
        struct.tree.gain = struct.tree.gain.at[node_id].set(gain)
        struct.tree.col = struct.tree.col.at[node_id].set(col)
        struct.tree.thr = struct.tree.thr.at[node_id].set(thr)
        struct.rank = struct.rank.at[node_id].set(rank)

        # l child
        struct.tree.score = struct.tree.score.at[l_child_id].set(score_l)
        struct.tree.gh_sum = struct.tree.gh_sum.at[l_child_id].set(gh_l)
        struct.rank_start = struct.rank_start.at[l_child_id].set(struct.rank_start[node_id])
        struct.rank_end = struct.rank_end.at[l_child_id].set(rank)
        struct.tree.value = struct.tree.value.at[l_child_id].set(self.get_leaf_value(gh_l))
        struct.tree.depth = struct.tree.depth.at[l_child_id].set(struct.tree.depth[node_id] + 1)

        # right child
        struct.tree.score = struct.tree.score.at[r_child_id].set(score_r)
        struct.tree.gh_sum = struct.tree.gh_sum.at[r_child_id].set(gh_r)
        struct.rank_start = struct.rank_start.at[r_child_id].set(rank)
        struct.rank_end = struct.rank_end.at[r_child_id].set(struct.rank_end[node_id])
        struct.tree.value = struct.tree.value.at[r_child_id].set(self.get_leaf_value(gh_r))
        struct.tree.depth = struct.tree.depth.at[l_child_id].set(struct.tree.depth[node_id] + 1)

        return struct

    def init_struct(self, gh, rank2index) -> Struct:
        gh_sum = jax.numpy.sum(gh, 0)
        struct = Struct.init(rank2index, self.num_nodes, gh.shape[-2])
        struct.tree.gh_sum = struct.tree.gh_sum.at[0].set(gh_sum)
        struct.tree.score = struct.tree.score.at[0].set(self.get_score(gh_sum))
        struct.tree.value = struct.tree.value.at[0].set(self.get_leaf_value(gh_sum))
        struct.tree.is_leaf = struct.tree.is_leaf.at[0].set(True)
        struct.first_free_id = 1
        struct.should_continue = self.num_leaves > 1
        return struct

    def best_split_fori(
            self,
            x,
            rank2index,
            gh,
            gh_sum,
            rank_start,
            rank_end,
            node_id=None
    ):
        def step(col, state):
            best_score_l, best_score_r, best_col, best_thr, best_rank, best_gh_l, best_gh_r = state

            score_l, score_r, thr, rank, gh_l, gh_r = self.best_split_column(
                x,
                rank2index,
                gh,
                gh_sum,
                col,
                rank_start,
                rank_end,
                node_id=node_id
            )

            improvement = (score_l + score_r) > (best_score_l + best_score_r)

            best_score_l = jax.numpy.where(improvement, score_l, best_score_l)
            best_score_r = jax.numpy.where(improvement, score_r, best_score_r)
            best_col = jax.numpy.where(improvement, col, best_col)
            best_thr = jax.numpy.where(improvement, thr, best_thr)
            best_rank = jax.numpy.where(improvement, rank, best_rank)
            best_gh_l = jax.numpy.where(improvement, gh_l, best_gh_l)
            best_gh_r = jax.numpy.where(improvement, gh_r, best_gh_r)

            state = best_score_l, best_score_r, best_col, best_thr, best_rank, best_gh_l, best_gh_r
            return state

        gh_r = best_gh_r = gh_sum
        gh_l = best_gh_l = jax.numpy.zeros_like(gh_r)
        best_score_l = self.get_score(gh_l)
        best_score_r = self.get_score(gh_r)
        best_thr = -jax.numpy.inf
        best_rank = rank_start
        best_col = jax.numpy.array(0, dtype=jax.numpy.uint32)
        state = best_score_l, best_score_r, best_col, best_thr, best_rank, best_gh_l, best_gh_r
        state = jax.lax.fori_loop(0, rank2index.shape[1], step, state)
        best_score_l, best_score_r, best_col, best_thr, best_rank, best_gh_l, best_gh_r = state
        return best_score_l, best_score_r, best_col, best_thr, best_rank, best_gh_l, best_gh_r

    def best_split_column(
            self,
            x,
            rank2index,
            gh,
            gh_sum,
            col,
            rank_start,
            rank_end,
            node_id=None
    ):
        gh_r = best_gh_r = gh_sum
        gh_l = best_gh_l = jax.numpy.zeros_like(gh_r)
        best_score_l = self.get_score(gh_l)
        best_score_r = self.get_score(gh_r)
        best_split = -jax.numpy.inf
        best_thr = rank_start
        fvalue = -jax.numpy.inf

        state = best_score_l, best_score_r, best_split, best_thr, best_gh_l, best_gh_r, gh_l, gh_r, fvalue

        def loop_step(rank, state):
            best_score_l, best_score_r, best_split, best_rank, best_gh_l, best_gh_r, gh_l, gh_r, prev_fvalue = state

            # gather data
            index = rank2index[rank, col]
            gh_i = gh[index, :, :]  # n_features, n_targets, 2
            fvalue = x[index, col]

            # check the split at 0.5 (fvalue + prev_fvalue)
            split = 0.5 * (fvalue + prev_fvalue)

            score_l = self.get_score(gh_l)
            score_r = self.get_score(gh_r)

            do_update = (score_l + score_r > best_score_l + best_score_r)
            do_update &= (split != fvalue)
            do_update &= (gh_l[..., 1].sum() >= self.min_child_weight)
            do_update &= (gh_r[..., 1].sum() >= self.min_child_weight)

            best_score_l = jax.lax.cond(do_update, lambda: score_l, lambda: best_score_l)
            best_score_r = jax.lax.cond(do_update, lambda: score_r, lambda: best_score_r)
            best_split = jax.lax.cond(do_update, lambda: split, lambda: best_split)
            best_rank = jax.lax.cond(do_update, lambda: rank, lambda: best_rank)
            best_gh_l = jax.lax.cond(do_update, lambda: gh_l, lambda: best_gh_l)
            best_gh_r = jax.lax.cond(do_update, lambda: gh_r, lambda: best_gh_r)

            # update with current observation statistics, will be used in the next iteration
            gh_l = gh_l + gh_i
            gh_r = gh_r - gh_i
            prev_fvalue = fvalue

            state = best_score_l, best_score_r, best_split, best_rank, best_gh_l, best_gh_r, gh_l, gh_r, prev_fvalue
            return state

        def loop_fn_while(state_while):
            rank, state = state_while
            state = loop_step(rank, state)
            return rank + 1, state

        state_while = rank_start, state

        state_while = jax.lax.while_loop(lambda arg: arg[0] < rank_end, loop_fn_while, state_while)

        _, (best_score_l, best_score_r, best_split, best_rank, best_gh_l, best_gh_r, _, _, _) = state_while
        return best_score_l, best_score_r, best_split, best_rank, best_gh_l, best_gh_r
