import jax
import dataclasses

from jax.typing import ArrayLike


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class GHTree:
    # generic
    depth: ArrayLike

    # split
    is_split: ArrayLike
    col: ArrayLike
    thr: ArrayLike
    gain: ArrayLike
    l_child_id: ArrayLike
    r_child_id: ArrayLike

    # leaf
    is_leaf: ArrayLike
    gh_sum: ArrayLike
    score: ArrayLike
    value: ArrayLike

    @classmethod
    def init(cls, num_nodes: int, num_targets: int) -> "GHTree":
        self = cls(
            depth=jax.numpy.zeros((num_nodes,), dtype=jax.numpy.uint32),

            is_split=jax.numpy.zeros((num_nodes,), dtype=jax.numpy.bool),
            col=jax.numpy.zeros((num_nodes,), dtype=jax.numpy.uint32),
            thr=jax.numpy.zeros((num_nodes,), dtype=jax.numpy.float32) + jax.numpy.nan,
            gain=jax.numpy.zeros((num_nodes,), dtype=jax.numpy.float32),
            l_child_id=jax.numpy.zeros((num_nodes,), dtype=jax.numpy.uint32),
            r_child_id=jax.numpy.zeros((num_nodes,), dtype=jax.numpy.uint32),

            is_leaf=jax.numpy.zeros((num_nodes,), dtype=jax.numpy.bool),
            gh_sum=jax.numpy.zeros((num_nodes, num_targets, 2), dtype=jax.numpy.float32) + jax.numpy.nan,
            score=jax.numpy.zeros((num_nodes,), dtype=jax.numpy.float32) + jax.numpy.nan,
            value=jax.numpy.zeros((num_nodes, num_targets), dtype=jax.numpy.float32) + jax.numpy.nan
        )
        return self

    def num_leaves(self):
        return jax.numpy.sum(self.is_leaf)

    def num_splits(self):
        return jax.numpy.sum(self.is_split)

    def predict_leaf_id(self, x):
        def _fn(state):
            i, xi = state
            go_left = xi[self.col[i]] <= self.thr[i]
            i = jax.numpy.where(go_left, self.l_child_id[i], self.r_child_id[i])
            return i, xi

        def _predict_one(xi):
            return jax.lax.while_loop(lambda state: self.is_split[state[0]], _fn, (0, xi))

        return jax.vmap(_predict_one)(x)[0]

    def predict_value(self, x):
        leaf_id = self.predict_leaf_id(x)
        value = self.value[leaf_id]
        return value

        # leaf_oh = jax.nn.one_hot(leaf_id, self.value.shape[0])
        # return jax.numpy.sum(jax.numpy.expand_dims(leaf_oh, -1) * jax.numpy.expand_dims(self.value, 0), 1)
