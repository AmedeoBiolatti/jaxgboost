import jax

from jaxgboost.tree_builders.base import TreeBuilder

from jaxgboost.trees.tree import GHTree


class Booster:
    def __init__(
            self,
            tree_builder: TreeBuilder,
            n_estimators: int = 100,
            learning_rate: float = 0.1,
            base_score: float | None = None,
            max_delta_step: float | None = None,
            colsample_bytree: float | None = None,
            subsample: float | None = None,
            num_parallel_trees: int = 1
    ):
        self.tree_builder = tree_builder
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_score = base_score
        self.max_delta_step = max_delta_step
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        self.num_parallel_trees = num_parallel_trees

    def _sample_data(self, x, y, p, sample_weight, aux_data, prng_key):
        if sample_weight is None:
            sample_weight = jax.numpy.ones(x.shape[:1])

        if self.colsample_bytree is not None:
            m = x.shape[0]
            prng_key, key_ = jax.random.split(prng_key)
            cols = jax.random.choice(key_, m, shape=(int((1 - self.colsample_bytree) * m),), replace=False)

            x = x.at[cols].set(jax.numpy.inf)

        if self.subsample is not None:
            sample_weight = (jax.random.uniform(prng_key, shape=x.shape[:1]) < self.subsample).astype(float)

        return x, y, p, sample_weight, aux_data

    def build_trees(
            self,
            x: jax.numpy.ndarray,
            y: jax.numpy.ndarray,
            sample_weight: jax.numpy.ndarray | None = None,
            *,
            aux_data=None,
            prng_key=None
    ):
        if aux_data is None:
            aux_data = self.tree_builder.get_aux_data(x, y=y)

        def build_one_tree(current_p, prng_key_):

            def build_one_tree_parallel(key_):
                x_, y_, current_p_, sample_weight_, aux_data_ = self._sample_data(
                    x, y, current_p, sample_weight, aux_data, key_
                )

                tree: GHTree = self.tree_builder.build_tree(
                    x_,
                    y_,
                    current_p_,
                    aux_data=aux_data_,
                    sample_weight=sample_weight_
                )
                tree_p = tree.predict_value(x_)

                tree_p = self.learning_rate * tree_p
                if self.max_delta_step is not None:
                    tree_p = jax.numpy.clip(tree_p, -self.max_delta_step, +self.max_delta_step)

                tree.value = self.learning_rate * tree.value
                if self.max_delta_step is not None:
                    tree.value = jax.numpy.clip(tree.value, -self.max_delta_step, +self.max_delta_step)

                return tree, tree_p

            tree, tree_p = jax.vmap(
                build_one_tree_parallel,
            )(jax.random.split(prng_key_, self.num_parallel_trees))

            tree.value = tree.value / self.num_parallel_trees
            tree_p = tree_p.mean(0)

            return current_p + tree_p, tree

        if self.base_score is None:
            base_score = jax.numpy.mean(y, axis=0, keepdims=True)
            base_score_squeezed = base_score[0]
        else:
            base_score = base_score_squeezed = self.base_score
        p0 = jax.numpy.zeros_like(y) + base_score
        p, trees = jax.lax.scan(
            build_one_tree,
            p0,
            None if prng_key is None else jax.random.split(prng_key, self.n_estimators),
            length=self.n_estimators if prng_key is None else None
        )

        trees.value = trees.value.at[0, :].add(base_score_squeezed)
        return trees, p

    def predict_values(self, trees, x):
        predict_values = GHTree.predict_value
        predict_values = jax.vmap(predict_values, (0, None))
        predict_values = jax.vmap(predict_values, (0, None))
        p = predict_values(trees, x)
        p = jax.numpy.sum(p, axis=(0, 1))
        return p
