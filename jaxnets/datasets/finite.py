"""`Dataset`s are sequences of unique examples."""
from typing import Any
from abc import ABC, abstractmethod
from collections.abc import Sequence
# from nptyping import NDArray
# from nptyping import Bool
# from nptyping import Floating
# from nptyping import Int
from numpy.typing import NDArray#, Bool, Floating, Int
from jax import Array

from enum import Enum
from enum import unique
from functools import cached_property
from functools import partial
import numpy as np
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.nn as jnn

from jaxnets.datasets.base import IndexType, ExemplarType, slice_to_array, Dataset

class FiniteDataset(Dataset):
  """A `Dataset` of finite class exemplars from which to draw sequences."""
  
  num_exemplars: int
  num_dimensions: int
  sequence_length: int
  inputs: Array
  labels: Array
  num_classes: int
  
  def __init__(
    self,
    key: Array,
    inputs: Array,
    labels: Array,
  ):
    """A `Dataset` of finite class exemplars from which to draw examples.

    Args:
      key: A key for randomness in sampling.
      inputs: A sequence of inputs.
      labels: A sequence of labels.
    """
    super().__init__(
      key=key,
      num_exemplars=len(labels),
      num_dimensions=inputs.shape[-1],
      sequence_length=0 if inputs.ndim < 3 else inputs.shape[1], # FIXME: this logic may not be correct
    )
    self.inputs = inputs
    self.labels = labels
    self.num_classes = len(np.unique(labels))
    
  def __getitem__(self, index: int | slice) -> ExemplarType:
    """Return an exemplar from the dataset."""
    if isinstance(index, (int, slice)):
      return self.inputs[index], self.labels[index]
    raise TypeError(f"Index type {type(index)} not supported.")