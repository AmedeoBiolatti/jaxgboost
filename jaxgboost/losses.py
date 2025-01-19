import jax
import chex


class Loss:
    def __init__(self):
        # self.grad = jax.grad(self.call, argnums=1)
        # self.hess = jax.vmap(jax.hessian(self.call, argnums=1))
        pass

    def call(self, y, p):
        raise NotImplementedError

    def __call__(self, y, p):
        chex.assert_equal(y.ndim, 2)
        chex.assert_equal(y.shape[1], 1)
        chex.assert_equal_shape((y, p))
        return self.call(y, p)

    def grad_and_hess(self, y, p):
        return self.grad(y, p), self.hess(y, p)


class MSE(Loss):
    def __init__(self):
        super().__init__()
        self.grad = lambda y, p: p - y
        self.hess = lambda y, p: jax.numpy.ones_like(y)

    def call(self, y, p):
        return jax.numpy.mean((y - p) ** 2)


def get(loss_name: str | Loss) -> Loss:
    if isinstance(loss_name, Loss):
        return loss_name
    if loss_name.lower() in ['l2', 'mse', 'reg:squarederror']:
        return MSE()
    raise ValueError(f"No loss called '{loss_name}'")
