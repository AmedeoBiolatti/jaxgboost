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
            min_child_weight: float = 0.0,
            num_leaves: int = -1
    ):
        self.loss = losses.get(objective)
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.min_split_loss = min_split_loss
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.num_leaves = num_leaves

    def build_tree(
            self,
            x: jax.numpy.ndarray,
            y: jax.numpy.ndarray,
            p: jax.numpy.ndarray | None = None,
            sample_weight: jax.numpy.ndarray | None = None,
            *,
            aux_data: dict[str, jax.numpy.ndarray] | None = None,
            **kwargs,
    ):
        raise NotImplementedError

    def get_leaf_value(self, gh):
        g, h = gh[..., 0], gh[..., 1]
        den = (self.reg_lambda + h)
        tmp = -g / den
        num = jax.numpy.where(
            tmp >= 0,
            -(g + self.reg_alpha),
            -(g - self.reg_alpha)
        )
        return num / den

    def get_score(self, gh):
        g, h = gh[..., 0], gh[..., 1]
        score = g ** 2 / (self.reg_lambda + h)
        return jax.numpy.sum(score, axis=-1)

    def init_data(self, x, y, sample_weight, p, aux_data):
        if p is None:
            p = jax.numpy.zeros_like(y)

        if aux_data is None:
            aux_data = self.get_aux_data(x, y=y)

        # init
        g, h = self.loss.grad_and_hess(y, p)
        gh = jax.numpy.stack((g, h), axis=-1)
        if sample_weight is not None:
            sample_weight = jax.numpy.reshape(sample_weight, (-1, 1, 1))
            gh = gh * sample_weight

        return x, y, sample_weight, p, aux_data, gh

    def get_aux_data(self, x, y=None):
        return {
            "rank2index": jax.numpy.argsort(x, axis=0)
        }
