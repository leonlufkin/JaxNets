"""Weight initializers for neural networks."""
import numpy as np
from math import sqrt

import jax
import jax.numpy as jnp
import jax.random as jrandom

import equinox as eqx
import equinox.nn as enn

from jax import Array
from collections.abc import Callable
import ipdb


def trunc_normal_init(
  weight: Array, 
  key: Array, 
  init_scale: float | None = None
) -> Array:
  """Truncated normal distribution initialization."""
  _, in_ = weight.shape
  init_scale = init_scale or sqrt(1.0 / max(1.0, in_))
  return init_scale * jax.random.truncated_normal(
    key=key,
    shape=weight.shape,
    lower=-2,
    upper=2,
  )

# Adapted from https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/initializers.py.
def lecun_normal_init(
  weight: Array,
  key: Array,
  init_scale: float = 1.0,
) -> Array:
  """LeCun (variance-scaling) normal distribution initialization."""
  _, in_ = weight.shape
  init_scale /= max(1.0, in_)

  stddev = np.sqrt(init_scale)
  # Adjust stddev for truncation.
  # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
  distribution_stddev = jnp.asarray(0.87962566103423978, dtype=float)
  stddev = stddev / distribution_stddev

  return trunc_normal_init(weight, key, init_scale=stddev)

def xavier_normal_init(
  weight: Array,
  key: Array,
  init_scale: float = 1.0,
) -> Array:
  xavier = jax.nn.initializers.glorot_normal()
  stddev = np.sqrt(init_scale)
  return stddev * xavier(key, weight.shape)
