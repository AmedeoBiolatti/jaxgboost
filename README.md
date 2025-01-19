# JAXGBoost

JAXGBoost is an extreme gradient boosting package completely implemented in JAX.
It tries to keep as much as possible of the interface the same of XGBoost/Lightgbm, but with the objective of being
completely interoperative with JAX.

### Basic usage

```python
model = jaxgboost.JAXGBoostModel()
model.fit(X, y)
pred = model.predict(X)
```

### Advanced

JAXGBoost is fully compatible with JAX functionality, such as `jax.jit` and/or `jax.vmap`

```python
@jax.jit
@jax.vmap
def fit_and_eval(params):
    model = jaxgboost.JAXGBoostModel(**params)
    model.fit(X_train, y_train)
    return jnp.mean((y_valid - model.predict(X_valid)) ** 2)


predictions = fit_and_eval({"learning_rate": jnp.linspace(0.01, 0.3, 10)})
print(predictions.shape)
# prints (10,)
```

### Roadmap

- [x] exact layer-wise tree building
- [ ] exact loss-guide tree building
