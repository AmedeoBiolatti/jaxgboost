

# JAXGBoost: Boosting de Gradiente para [JAX](https://github.com/jax-ml/jax)

JAXGBoost es una biblioteca de Python que implementa máquinas de boosting de gradiente en JAX puro.

Busca mantener la interfaz lo más similar posible a XGBoost/LightGBM, mientras ofrece una interoperabilidad perfecta con el ecosistema de JAX.

### Instalación

```
pip install git+https://github.com/AmedeoBiolatti/jaxgboost
```

### Inicio rápido

```python
import jaxgboost

# carga tus datos de entrenamiento y prueba
(X_train, y_train), (X_valid, y_valid) = load_dataset()

# Crea y entrena el modelo
model = jaxgboost.JAXGBoostModel()
model.fit(X_train, y_train)

# Realiza predicciones
pred = model.predict(X_valid)
```

### Uso avanzado

JAXGBoost admite funcionalidades de JAX como `jit` y `vmap`. Aquí tienes un ejemplo:

```python
import jax


# Crea una función vectorizada con jit
@jax.jit
@jax.vmap
def fit_and_eval(params):
    model = jaxgboost.JAXGBoostModel(**params)
    model.fit(X_train, y_train)
    return jnp.mean((y_valid - model.predict(X_valid)) ** 2)


# Evalúa la función con 10 valores de parámetros al mismo tiempo
mse_values = fit_and_eval({"learning_rate": jnp.linspace(0.01, 0.3, 10)})
print(mse_values.shape)
# imprime (10,)
```

### Hoja de ruta

- [x] construcción exacta de árboles por capas
- [x] construcción de árboles guiada por la pérdida
- [ ] construcción de árboles basada en histogramas
- [ ] implementación compatible con gradientes para optimización hiperparamétrica
- [ ] predicción "suavizada" para una mejor integración en redes neuronales
- [ ] registro (logging)
