import chex
import jax

from jaxgboost import losses
from jaxgboost.tree_builders.base import TreeBuilder


class ExactLayerWiseVMapTreesBuilder(TreeBuilder):
    def __init__(
            self,
            objective: str | losses.Loss = 'l2',
            reg_lambda: float = 1.0,
            reg_alpha: float = 0.0,
            min_split_loss: float = 0.0,
            max_depth: int = 6,
            min_child_weight: float = 0.0
    ):
        super().__init__(objective=objective, reg_lambda=reg_lambda, reg_alpha=reg_alpha, min_split_loss=min_split_loss)
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight

    def get_aux_data(
            self,
            x,
            y=None
    ) -> dict[str, jax.numpy.ndarray]:
        rank2index = jax.numpy.argsort(x, axis=0)
        return {
            'rank2index': rank2index,
        }

    def build_tree(
            self,
            x: jax.numpy.ndarray,
            y: jax.numpy.ndarray,
            p: jax.numpy.ndarray | None = None,
            sample_weight: jax.numpy.ndarray | None = None,
            *,
            aux_data: dict[str, jax.numpy.ndarray] | None = None,
            **kwargs,
    ) -> tuple["Tree", jax.numpy.ndarray]:
        chex.assert_equal(x.shape[0], y.shape[0])
        if sample_weight is not None:
            chex.assert_equal(x.shape[0], p.shape[0])
        if sample_weight is not None:
            chex.assert_equal(x.shape[0], sample_weight.shape[0])

        # setup
        n_obs, n_features = x.shape
        num_leaves = 2 ** self.max_depth

        if p is None:
            p = jax.numpy.zeros_like(y)

        if aux_data is None:
            aux_data = self.get_aux_data(x, y=y)

        rank2index = aux_data['rank2index']

        # init
        g, h = self.loss.grad_and_hess(y, p)
        gh = jax.numpy.stack((g, h), axis=-1)
        if sample_weight is not None:
            sample_weight = jax.numpy.reshape(sample_weight, (-1, 1, 1))
            gh = gh * sample_weight

        position = jax.numpy.zeros((n_obs,), dtype=int)

        gh_sum = jax.numpy.zeros((num_leaves, gh.shape[1], 2)).at[0].set(gh.sum(0))

        def get_split_at_depth(state, _):
            position, gh_sum = state

            best_score, best_col, best_split, best_gh_l, best_gh_r = self._get_splits_at_level(
                x,
                gh,
                gh_sum,
                position,
                rank2index
            )
            x_ref = jax.numpy.take(x, jax.numpy.arange(n_obs) * n_features + best_col[position])
            mask = (x_ref <= best_split[position]).astype(int)
            position = 2 * position + mask

            # b_gh_r = gh_sum - b_gh_l
            gh_sum = gh_sum.at[1::2].set(best_gh_l[:gh_sum.shape[0] // 2])
            gh_sum = gh_sum.at[::2].set(best_gh_r[:gh_sum.shape[0] // 2])

            state = position, gh_sum
            return state, (best_col, best_split)

        (position, _), splits = jax.lax.scan(
            get_split_at_depth,
            (position, gh_sum),
            None,
            length=self.max_depth
        )

        leaf_one_hot = jax.nn.one_hot(position, num_leaves).reshape(n_obs, num_leaves, 1)
        g_leaves = jax.numpy.sum(leaf_one_hot * jax.numpy.expand_dims(g, 1), axis=0)
        h_leaves = jax.numpy.sum(leaf_one_hot * jax.numpy.expand_dims(h, 1), axis=0)

        values = self._get_leaves(g_leaves, h_leaves)
        p = jax.numpy.sum(jax.numpy.expand_dims(values, 0) * leaf_one_hot, axis=1)

        tree = splits, values
        return tree, p

    def _get_splits_at_level(
            self,
            x,
            gh,
            gh_sum,
            position,
            rank2index
    ):
        n_cols = x.shape[1]

        def loop_step(col):
            score, split, gh_l, gh_r = self._get_splits_at_level_on_column(
                col,
                x,
                gh,
                gh_sum,
                position,
                rank2index
            )
            return score, split, gh_l, gh_r

        score_, split_, gh_l_, gh_r_ = jax.vmap(loop_step)(jax.numpy.arange(n_cols))
        best_col = jax.numpy.argmax(score_, axis=0)

        best_score = jax.numpy.stack([score_[c, i] for i, c in enumerate(best_col)])
        best_split = jax.numpy.stack([split_[c, i] for i, c in enumerate(best_col)])
        best_gh_l = jax.numpy.stack([gh_l_[c, i] for i, c in enumerate(best_col)])
        best_gh_r = jax.numpy.stack([gh_r_[c, i] for i, c in enumerate(best_col)])
        return best_score, best_col, best_split, best_gh_l, best_gh_r

    def _get_splits_at_level_on_column(
            self,
            col,
            x,
            gh,
            gh_sum,
            position,
            rank2index
    ):
        n_obs = x.shape[0]
        num_leaves = gh_sum.shape[0]

        def loop_step(rank, state):
            (best_score, best_split, best_gh_l, best_gh_r), (gh_l, gh_r), (prev_fvalue,) = state

            # gather data
            index = rank2index[rank, col]
            pos = position[index]
            gh_i = gh[index, :, :]  # n_features, n_targets, 2
            fvalue = x[index, col]

            # check the split at 0.5 (fvalue + prev_fvalue)
            split = 0.5 * (fvalue + prev_fvalue[pos])
            split_score = self._get_score(gh_l[pos]) + self._get_score(gh_r[pos])

            do_update = (split_score > best_score[pos])
            do_update &= (split != fvalue)
            do_update &= (gh_l[pos, ..., 1].sum() >= self.min_child_weight)
            do_update &= (gh_r[pos, ..., 1].sum() >= self.min_child_weight)

            best_score = jax.lax.cond(do_update, lambda: best_score.at[pos].set(split_score), lambda: best_score)
            best_split = jax.lax.cond(do_update, lambda: best_split.at[pos].set(split), lambda: best_split)
            best_gh_l = jax.lax.cond(do_update, lambda: best_gh_l.at[pos].set(gh_l[pos]), lambda: best_gh_l)
            best_gh_r = jax.lax.cond(do_update, lambda: best_gh_r.at[pos].set(gh_r[pos]), lambda: best_gh_r)

            # update with current observation statistics, will be used in the next iteration
            gh_l = gh_l.at[pos].add(gh_i)
            gh_r = gh_r.at[pos].add(-gh_i)
            prev_fvalue = prev_fvalue.at[pos].set(fvalue)

            state = (best_score, best_split, best_gh_l, best_gh_r), (gh_l, gh_r), (prev_fvalue,)
            return state

        best_score = jax.numpy.zeros((num_leaves,))
        best_split = jax.numpy.zeros((num_leaves,)) - jax.numpy.inf
        best_gh_l = gh_l = jax.numpy.zeros_like(gh_sum)
        best_gh_r = gh_r = gh_sum
        prev_fvalue = jax.numpy.zeros((num_leaves,)) - jax.numpy.inf
        state = (
            (best_score, best_split, best_gh_l, best_gh_r), (gh_l, gh_r), (prev_fvalue,)
        )
        state = jax.lax.fori_loop(
            0,
            n_obs,
            loop_step,
            state
        )
        (best_score, best_split, best_gh_l, best_gh_r), _, _ = state
        return best_score, best_split, best_gh_l, best_gh_r
