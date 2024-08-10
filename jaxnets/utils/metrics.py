"""
Admin utilities
labeling (that's it for now)

Author: Leon Lufkin
Last modified: 2024-08-08
"""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from collections.abc import Generator
from jax import Array

import jax
import jax.numpy as jnp

import ipdb



def accuracy(pred_y: Array, y: Array) -> Array:
  """Compute elementwise accuracy."""
  predicted_class = jnp.where(pred_y > 0.5, 1., 0.)
  return predicted_class == y

def mse(pred_y: Array, y: Array) -> Array:
  """Compute elementwise mean squared error."""
  return jnp.square(pred_y - y)

def ce(pred_y: Array, y: Array) -> Array:
  """Compute elementwise cross-entropy loss."""
  pred_y = jnp.exp(pred_y) / jnp.sum(jnp.exp(pred_y), axis=-1, keepdims=True)
  y = jax.nn.one_hot(y, pred_y.shape[-1])
  return -jnp.sum(y * jnp.log(pred_y), axis=-1)