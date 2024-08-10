"""Simple feedforward neural networks."""
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

from jaxnets.models.initializers import trunc_normal_init, lecun_normal_init, xavier_normal_init

class StopGradient(eqx.Module):
  """Stop gradient wrapper."""

  array: jnp.ndarray

  def __jax_array__(self):
    """Return the array wrapped with a stop gradient op."""
    return jax.lax.stop_gradient(self.array)


class Linear(enn.Linear):
  """Linear layer."""

  weight: Array
  bias: Array

  def __init__(
    self,
    in_size: int,
    out_size: int,
    use_bias: bool = True,
    weight_trainable: bool = True,
    bias_value: float = 0.0,
    bias_trainable: bool = False,
    *,
    key: Array,
    init_fn: Callable = xavier_normal_init,
    **init_kwargs,
  ):
    """Initialize a linear layer."""
    super().__init__(
      in_features=in_size,
      out_features=out_size,
      use_bias=use_bias,
      key=key,
    )

    # Reinitialize weight from variance scaling distribution, reusing `key`.
    self.weight: Array = init_fn(self.weight, key=key, **init_kwargs)
    if not weight_trainable:
      self.weight = StopGradient(self.weight)

    # Reinitialize bias to zeros.
    if use_bias:
      self.bias: Array = bias_value * jnp.ones_like(self.bias)

      if not bias_trainable:
        self.bias = StopGradient(self.bias)


class MLP(eqx.Module):
  """Multi-layer perceptron."""

  fc1: eqx.Module
  activation: Callable
  fc2: eqx.Module
  num_hiddens: int

  def __init__(
    self,
    in_size: int,
    hidden_size: int | None = None,
    out_size: int | None = 1,
    activation: Callable = lambda x: x,
    *,
    key: Array = None,
    init_fn: Callable = xavier_normal_init,
    **linear_kwargs,
  ):
    """Initialize an MLP.

    Args:
       in_size: The expected dimension of the input.
       hidden_size: Dimensionality of the hidden layer.
       out_size: The dimension of the output feature.
       activation: Activation function to be applied to the intermediate layers.
       drop: The probability associated with `Dropout`.
       key: A `jax.random.PRNGKey` used to provide randomness for parameter
        initialisation.
       init_scale: The scale of the variance of the initial weights.
    """
    super().__init__()
    out_size = out_size or in_size
    hidden_size = hidden_size or in_size
    key1, key2 = jrandom.split(key, 2)

    self.fc1 = Linear(
      in_size=in_size,
      out_size=hidden_size,
      key=key1,
      init_fn=init_fn,
      **linear_kwargs,
    )
    self.activation = activation
    self.fc2 = Linear(
      in_size=hidden_size,
      out_size=out_size,
      key=key2,
      init_fn=init_fn,
      **linear_kwargs,
    )
    self.num_hiddens = hidden_size

  def forward_pass(self, x: Array, *, key: Array) -> Array:
    preact = self.fc1(x)
    x = self.activation(preact)
    x = self.fc2(x) / self.num_hiddens
    return x, preact

  def __call__(self, x: Array, *, key: Array) -> Array:
    return self.forward_pass(x, key=key)[0]


class SCM(eqx.Module):
  """
  Soft-Committee Machine, i.e. a two-layer MLP with second layer weights fixed so they take the average. 
  By construction, the SCM has 1D output.
  """

  fc1: eqx.Module
  activation: Callable

  def __init__(
    self,
    in_size: int,
    hidden_size: int | None = None,
    activation: Callable = lambda x: x,
    *,
    key: Array = None,
    init_fn: Callable = xavier_normal_init,
    **linear_kwargs
  ):
    """Initialize an SCM.

    Args:
      in_size: The expected dimension of the input.
      hidden_size: Dimensionality of the hidden layer.
      out_size: The dimension of the output feature.
      activation: Activation function to be applied to the intermediate layers.
      drop: The probability associated with `Dropout`.
      key: A `jax.random.PRNGKey` used to provide randomness for parameter
      initialisation.
      init_scale: The scale of the variance of the initial weights.
    """
    super().__init__()
    hidden_size = hidden_size or in_size
    linear_kwargs.pop('out_size')

    self.fc1 = Linear(
      in_size=in_size,
      out_size=hidden_size,
      key=key,
      init_fn=init_fn,
      **linear_kwargs # TODO: try use_bias = False
    ) 
    self.activation = activation

  def forward_pass(self, x: Array, *, key: Array) -> Array:
    """Apply the MLP block to the input."""
    preact = self.fc1(x) # first layer
    x = self.activation(preact)
    x = jnp.mean(x) # second layer
    return x, preact

  def __call__(self, x: Array, *, key: Array) -> Array:
    return self.forward_pass(x, key=key)[0]


class GatedNet(eqx.Module):
  """
  SCM, but rather than an activation, we apply a gating function.
  """

  fc1: eqx.Module
  gate: Callable

  def __init__(
    self,
    in_size: int,
    hidden_size: int | None = None,
    gate: Callable = lambda x: 1.,
    *,
    key: Array = None,
    init_fn: Callable = xavier_normal_init,
    **linear_kwargs
  ):
    """Initialize a GatedNet, but an SCM for now.

    Args:
       in_size: The expected dimension of the input.
       hidden_size: Dimensionality of the hidden layer.
       out_size: The dimension of the output feature.
       activation: Gating function to be applied to the intermediate layer.
            Given input x, returns a vector of gates for all the intermediate neurons.
       drop: The probability associated with `Dropout`.
       key: A `jax.random.PRNGKey` used to provide randomness for parameter
        initialisation.
       init_scale: The scale of the variance of the initial weights.
    """
    super().__init__()
    hidden_size = hidden_size or in_size
    
    Warning("GatedNet is currently just an SCM with a gating function.")

    self.fc1 = Linear(
      in_size=in_size,
      out_size=hidden_size,
      key=key,
      init_fn=init_fn,
      **linear_kwargs # TODO: try use_bias = False
    ) 
    self.gate = gate


  def forward_pass(self, x: Array, *, key: Array) -> Array:
    """Apply the MLP block to the input."""
    preacts = self.fc1(x)
    gates = self.gate(x)
    postacts = gates * preacts
    x = jnp.mean(postacts)
    return x, preacts, gates, postacts
  
  def __call__(self, x: Array, *, key: Array) -> Array:
    return self.forward_pass(x, key=key)[0]