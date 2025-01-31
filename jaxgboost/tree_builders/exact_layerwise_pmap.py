import chex
import jax

from jaxgboost import losses
from jaxgboost.tree_builders.exact_layerwise import ExactLayerWiseTreesBuilder
from jaxgboost.trees.tree import GHTree


class ExactLayerWisePMapTreesBuilder(ExactLayerWiseTreesBuilder):
    def __init__(
            self,
            objective: str | losses.Loss = 'l2',
            reg_lambda: float = 1.0,
            reg_alpha: float = 0.0,
            min_split_loss: float = 0.0,
            max_depth: int = 6,
            min_child_weight: float = 0.0
    ):
        super().__init__(
            objective=objective,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            min_split_loss=min_split_loss,
            max_depth=max_depth,
            min_child_weight=min_child_weight
        )
        self.n_devices = len(jax.devices())

    def _get_splits_at_level(
            self,
            x,
            gh,
            gh_sum,
            position,
            rank2index
    ):
        n_cols = x.shape[1]
        num_leaves = 2 ** self.max_depth

        def loop_step(state, col):
            best_score, best_col, best_split, best_gh_l, best_gh_r = state

            score, split, gh_l, gh_r = self._get_splits_at_level_on_column(
                col,
                x,
                gh,
                gh_sum,
                position,
                rank2index
            )

            update_mask = (score > best_score)
            update_mask_ = jax.numpy.reshape(update_mask, (-1, 1, 1))

            best_score = jax.numpy.where(update_mask, score, best_score)
            best_col = jax.numpy.where(update_mask, col, best_col)
            best_split = jax.numpy.where(update_mask, split, best_split)
            best_gh_l = jax.numpy.where(update_mask_, gh_l, best_gh_l)
            best_gh_r = jax.numpy.where(update_mask_, gh_r, best_gh_r)

            state = best_score, best_col, best_split, best_gh_l, best_gh_r
            return state, ()

        if n_cols < self.n_devices:
            cols = jax.numpy.arange(n_cols)
            cols = jax.numpy.reshape(cols, (1, -1))
        elif n_cols % self.n_devices == 0:
            cols = jax.numpy.arange(n_cols)
            cols = jax.numpy.reshape(cols, (-1, self.n_devices))
        else:
            mod = n_cols % self.n_devices
            cols = jax.numpy.arange(n_cols + self.n_devices - mod)
            cols = jax.numpy.reshape(cols, (-1, self.n_devices))
        cols = jax.numpy.transpose(cols)

        def fn(cols):
            best_score = self._get_score(gh_sum) + self.min_split_loss
            best_col = jax.numpy.zeros((num_leaves,), dtype=jax.numpy.int32)
            best_split = jax.numpy.zeros((num_leaves,)) - jax.numpy.inf
            best_gh_l = jax.numpy.zeros_like(gh_sum)
            best_gh_r = gh_sum
            state = best_score, best_col, best_split, best_gh_l, best_gh_r
            state, _ = jax.lax.scan(
                loop_step,
                state,
                cols
            )
            best_score, best_col, best_split, best_gh_l, best_gh_r = state
            return best_score, best_col, best_split, best_gh_l, best_gh_r

        score_, col_, split_, gh_l_, gh_r_ = jax.pmap(fn)(cols)

        best_i = jax.numpy.argmax(score_, axis=0)
        best_score = jax.numpy.stack([score_[c, i] for i, c in enumerate(best_i)])
        best_col = jax.numpy.stack([col_[c, i] for i, c in enumerate(best_i)])
        best_split = jax.numpy.stack([split_[c, i] for i, c in enumerate(best_i)])
        best_gh_l = jax.numpy.stack([gh_l_[c, i] for i, c in enumerate(best_i)])
        best_gh_r = jax.numpy.stack([gh_r_[c, i] for i, c in enumerate(best_i)])
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

            def do_update_fn():
                return (
                    best_score.at[pos].set(split_score),
                    best_split.at[pos].set(split),
                    best_gh_l.at[pos].set(gh_l[pos]),
                    best_gh_r.at[pos].set(gh_r[pos])
                )

            best_score, best_split, best_gh_l, best_gh_r = jax.lax.cond(
                do_update,
                do_update_fn,
                lambda: (best_score, best_split, best_gh_l, best_gh_r)
            )

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
