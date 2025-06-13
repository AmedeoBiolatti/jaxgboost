import unittest

import numpy as np
import xgboost

import jaxgboost


def get_data():
    np.random.seed(1)
    x = np.random.normal(size=(100, 2))
    y = x[:, 1].reshape(100, 1)
    return x, y


class SKLearnCase(unittest.TestCase):
    def test_same_results_xgboost(self):
        x, y = get_data()

        model_xgboost = xgboost.XGBRegressor(
            n_estimators=20, max_depth=4, tree_method="exact", base_score=y.mean()
        ).fit(x, y)
        model_jaxgboost = jaxgboost.JAXGBoostModel(
            n_estimators=20,
            max_depth=4,
            learning_rate=0.3,
            base_score=y.mean()
        ).fit(x, y)

        np.testing.assert_allclose(
            model_jaxgboost.predict(x).ravel(),
            model_xgboost.predict(x).ravel(),
            atol=1e-6,
            rtol=1e-6
        )
