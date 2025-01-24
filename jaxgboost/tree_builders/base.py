import jax

from jaxgboost import losses


class TreeBuilder:
    def __init__(
            self,
            objective: str | losses.Loss = 'l2',
            reg_lambda: float = 1.0,
            reg_alpha: float = 0.0,
            min_split_loss: float = 0.0,
            max_depth: int = 6,
            min_child_weight: float = 0.0
    ):
        self.loss = losses.get(objective)
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.min_split_loss = min_split_loss
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight

    def _get_leaves(self, g, h) -> jax.numpy.ndarray:
        den = (self.reg_lambda + h)
        tmp = -g / den
        num = jax.numpy.where(
            tmp >= 0,
            -(g + self.reg_alpha),
            -(g - self.reg_alpha)
        )
        return num / den

    def _get_score(self, gh):
        g, h = gh[..., 0], gh[..., 1]
        score = g ** 2 / (self.reg_lambda + h)
        return jax.numpy.sum(score, axis=-1)

    def predict_leaf_id(
            self,
            tree: "Tree",
            x: jax.numpy.ndarray
    ) -> jax.numpy.ndarray:
        def update_position(position_, c_t):
            cols, thrs = c_t
            x_ref = jax.numpy.take(x, jax.numpy.arange(n_obs) * n_features + cols[position_])
            mask = (x_ref <= thrs[position_]).astype(int)
            position_ = 2 * position_ + mask
            return position_, None

        splits, values = tree

        n_obs, n_features = x.shape

        position = jax.numpy.zeros((n_obs,), dtype=int)
        position, _ = jax.lax.scan(update_position, position, splits)
        return position

    def predict_values(
            self,
            tree: "Tree",
            x: jax.numpy.ndarray
    ) -> jax.numpy.ndarray:
        _, values = tree
        num_leaves, num_targets = values.shape
        n_obs, n_features = x.shape

        position = self.predict_leaf_id(tree, x)
        leaf_one_hot = jax.nn.one_hot(position, num_leaves).reshape(n_obs, num_leaves, 1)
        p = jax.numpy.sum(jax.numpy.expand_dims(values, 0) * leaf_one_hot, 1)
        return p

    def update_leaves(
            self,
            tree: "Tree",
            x: jax.numpy.ndarray,
            y: jax.numpy.ndarray,
            p: jax.numpy.ndarray | None = None,
            sample_weight: jax.numpy.ndarray | None = None,
            **kwargs
    ) -> tuple["Tree", jax.numpy.ndarray]:
        splits, values = tree
        num_leaves, num_targets = values.shape
        n_obs, n_features = x.shape

        if p is None:
            p = jax.numpy.zeros_like(y)

        g, h = self.loss.grad_and_hess(y, p)

        position = self.predict_leaf_id(tree, x)
        leaf_one_hot = jax.nn.one_hot(position, num_leaves).reshape(n_obs, num_leaves, 1)
        g_leaves = jax.numpy.sum(leaf_one_hot * jax.numpy.expand_dims(g, 1), 0)
        h_leaves = jax.numpy.sum(leaf_one_hot * jax.numpy.expand_dims(h, 1), 0)

        values = self._get_leaves(g_leaves, h_leaves)

        tree_p = jax.numpy.sum(jax.numpy.expand_dims(values, 0) * leaf_one_hot, 1)

        tree = splits, values
        return tree, tree_p
