# JAXGBoost: Gradient Boosting for [JAX](https://github.com/jax-ml/jax)

JAXGBoost is a Python library implementing gradient boosting machines in pure JAX.

It aims to keep the interface as similar as possible to XGBoost/LightGBM, while offering seamless interoperability with
JAX ecosystem.

### Installation

```
pip install git+https://github.com/AmedeoBiolatti/jaxgboost
```

### Quickstart

```python
import jaxgboost

# load your training and test data
(X_train, y_train), (X_valid, y_valid) = load_dataset()

# Create and train the model
model = jaxgboost.JAXGBoostModel()
model.fit(X_train, y_train)

# Make predictions
pred = model.predict(X_valid)
```

### Advanced usage

JAXGBoost supports JAX functionalities such as `jit` and `vmap`. Here's an example:

```python
import jax

# Create a jitted vectorized function
@jax.jit
@jax.vmap
def fit_and_eval(params):
    model = jaxgboost.JAXGBoostModel(**params)
    model.fit(X_train, y_train)
    return jnp.mean((y_valid - model.predict(X_valid)) ** 2)


# Evaluate the function on 10 parameters values at the same time
mse_values = fit_and_eval({"learning_rate": jnp.linspace(0.01, 0.3, 10)})
print(mse_values.shape)
# prints (10,)
```

### Roadmap

- [x] exact layer-wise tree building
- [ ] loss-guide tree building
- [ ] hist tree building
- [ ] gradient-friendly implementation for hyper-opt
- [ ] "softened" prediction for better integration in NN