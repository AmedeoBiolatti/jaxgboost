import jax
import keras

from jaxgboost.interfaces.xgboost_import import from_xgboost


class VariableGHTree(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def from_xgboost(self, model):
        self.ghtree_ = from_xgboost(model)
        self.predict_fn_ = jax.vmap(jax.vmap(self.ghtree_.__class__.predict_value, (0, None)), (0, None))

    def build(self, input_shape):
        value = self.ghtree_.value
        self.value = self.add_weight(shape=value.shape, initializer=keras.initializers.Constant(value))

    def call(self, inputs):
        self.ghtree_.value = self.value.value
        return self.predict_fn_(self.ghtree_, inputs).sum((0, 1)).reshape((inputs.shape[0], -1))

    def compute_output_shape(self, input_shape):
        vshape = self.ghtree_.value.shape
        if len(vshape) == 3:
            return input_shape[:-1] + (1,)
        return input_shape[:-1] + (vshape[-1],)
