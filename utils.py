import jax.numpy as jnp


def relu(x):
    return jnp.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def tanh(x):
    return jnp.tanh(x)

def logit(y):
    if y <= 0 or y >= 1:
        raise ValueError("Input must be in the range (0, 1), exclusive.")
    return jnp.log(y / (1 - y))

