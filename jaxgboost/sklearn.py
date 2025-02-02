import jax

from jaxgboost import booster, tree_builders


def get_tree_builder(
        grow_policy
):
    if grow_policy.lower() in ["depthwise", "layerwise"]:
        if len(jax.devices()) > 1:
            return tree_builders.ExactLayerWisePMapTreesBuilder
        else:
            return tree_builders.ExactLayerWiseTreesBuilder
    elif grow_policy.lower() in ["lossguide"]:
        return tree_builders.LossGuideTreeBuilder
    else:
        raise ValueError(f"No known grow_policy named '{grow_policy}'")


class JAXGBoostModel:
    def __init__(
            self,
            objective: str = "mse",
            n_estimators: int = 100,
            learning_rate: float = 0.3,
            base_score: float | None = None,
            max_delta_step: float | None = None,
            colsample_bytree: float | None = None,
            subsample: float | None = None,
            num_parallel_trees: int = 1,
            reg_lambda: float = 1.0,
            reg_alpha: float = 0.0,
            min_split_loss: float = 0.0,
            max_depth: int = 6,
            min_child_weight: float = 0.0,
            grow_policy: str = "depthwise",
            num_leaves=None,
            random_state: int = 42,
            jit: bool = True
    ):
        self.tree_builder = get_tree_builder(
            grow_policy=grow_policy
        )(
            objective=objective,
            max_depth=max_depth,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            min_split_loss=min_split_loss,
            min_child_weight=min_child_weight,
            num_leaves=num_leaves
        )
        self.booster = booster.Booster(
            self.tree_builder,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            base_score=base_score,
            max_delta_step=max_delta_step,
            colsample_bytree=colsample_bytree,
            subsample=subsample,
            num_parallel_trees=num_parallel_trees
        )
        self.jit = jit
        self._predict_fn = self.booster.predict_values
        if self.jit:
            self._predict_fn = jax.jit(self._predict_fn)
        self._build_trees = self.booster.build_trees
        if self.jit:
            self._build_trees = jax.jit(self._build_trees)
        self.prng_key = jax.random.PRNGKey(random_state)

    def fit(self, X, y, sample_weight=None):
        self.trees_, _ = self._build_trees(X, y, sample_weight=sample_weight, prng_key=self.prng_key)
        return self

    def predict(self, X):
        predict = self._predict_fn(self.trees_, X)
        return predict
