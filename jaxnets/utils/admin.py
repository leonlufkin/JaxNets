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

import math

import ipdb

class Ignore:
  def __init__(self, value):
    self._value = value

  def __getattribute__(self, name):
    if name == '_value':
      return object.__getattribute__(self, '_value')
    value = object.__getattribute__(self, '_value')
    return getattr(value, name)

  def __repr__(self):
    return f"Ignore({repr(self._value)})"

  def unwrap(self):
    return self._value

  # Arithmetic operations
  def __add__(self, other): return self._value + other
  def __radd__(self, other): return other + self._value
  def __sub__(self, other): return self._value - other
  def __rsub__(self, other): return other - self._value
  def __mul__(self, other): return self._value * other
  def __rmul__(self, other): return other * self._value
  def __truediv__(self, other): return self._value / other
  def __rtruediv__(self, other): return other / self._value
  def __floordiv__(self, other): return self._value // other
  def __rfloordiv__(self, other): return other // self._value
  def __mod__(self, other): return self._value % other
  def __rmod__(self, other): return other % self._value
  def __divmod__(self, other): return divmod(self._value, other)
  def __rdivmod__(self, other): return divmod(other, self._value)
  def __pow__(self, other): return self._value ** other
  def __rpow__(self, other): return other ** self._value

  # Bitwise operations
  def __lshift__(self, other): return self._value << other
  def __rlshift__(self, other): return other << self._value
  def __rshift__(self, other): return self._value >> other
  def __rrshift__(self, other): return other >> self._value
  def __and__(self, other): return self._value & other
  def __rand__(self, other): return other & self._value
  def __xor__(self, other): return self._value ^ other
  def __rxor__(self, other): return other ^ self._value
  def __or__(self, other): return self._value | other
  def __ror__(self, other): return other | self._value

  # Unary operations
  def __neg__(self): return -self._value
  def __pos__(self): return +self._value
  def __abs__(self): return abs(self._value)
  def __invert__(self): return ~self._value

  # Type conversion
  def __complex__(self): return complex(self._value)
  def __int__(self): return int(self._value)
  def __float__(self): return float(self._value)
  def __round__(self, n=None): return round(self._value, n)
  def __trunc__(self): return math.trunc(self._value)
  def __floor__(self): return math.floor(self._value)
  def __ceil__(self): return math.ceil(self._value)

  # Comparison operations
  def __lt__(self, other): return self._value < other
  def __le__(self, other): return self._value <= other
  def __eq__(self, other): return self._value == other
  def __ne__(self, other): return self._value != other
  def __gt__(self, other): return self._value > other
  def __ge__(self, other): return self._value >= other

  # Container operations
  def __len__(self): return len(self._value)
  def __getitem__(self, key): return self._value[key]
  def __setitem__(self, key, value): self._value[key] = value
  def __delitem__(self, key): del self._value[key]
  def __iter__(self): return iter(self._value)
  def __reversed__(self): return reversed(self._value)
  def __contains__(self, item): return item in self._value

  # String representation
  def __str__(self): return str(self._value)
  def __bytes__(self): return bytes(self._value)
  def __format__(self, format_spec): return format(self._value, format_spec)

  # Context manager
  def __enter__(self): return self._value.__enter__()
  def __exit__(self, exc_type, exc_value, traceback): return self._value.__exit__(exc_type, exc_value, traceback)

  # Descriptor protocol
  def __get__(self, instance, owner): return self._value.__get__(instance, owner)
  def __set__(self, instance, value): self._value.__set__(instance, value)
  def __delete__(self, instance): self._value.__delete__(instance)

  # Pickling
  def __reduce__(self): return (Ignore, (self._value,))
  def __reduce_ex__(self, protocol): return self.__reduce__()

  # Hashing
  def __hash__(self): return hash(self._value)

  # Async operations
  def __await__(self): return self._value.__await__()
  def __aiter__(self): return self._value.__aiter__()
  def __anext__(self): return self._value.__anext__()

  # Other special methods
  def __bool__(self): return bool(self._value)
  def __call__(self, *args, **kwargs): return self._value(*args, **kwargs)
  
  

def make_key(
  *,
  remove_keys: list = ['wandb_', 'save_weights', 'save_model', 'task', 'config_modifier'],
  remove_none: bool = True,
  remove_dict: bool = True,
  key_prefixes: list = ['num_', 'use_'],
  key_suffixes: list = ['_cls', '_fn', '_size'],
  value_prefixes: list = ['simulate_'],
  value_suffixes: list = ['_init', 'Sampler'],
  config: dict = {},
):
  # make a copy of the config
  config = config.copy()
  # remove specific unwanted keys
  for k in remove_keys:
    if k in config:
      config.pop(k)
  # remove keys with None values, if desired
  if remove_none:
    for k in list(config.keys()):
      if config[k] is None:
        config.pop(k)
  # remove keys where the value is a dict, if desired
  if remove_dict:
    for k in list(config.keys()):
      if isinstance(config[k], dict):
        config.pop(k)
  # get __name__ of classes and functions
  for k, v in config.items():
    config[k] = getattr(v, '__name__', v)
  # collapse all tuple/list/array values to strings
  for k, v in config.items():
    if isinstance(v, (tuple, list, Array)):
      config[k] = ','.join([str(x) for x in v])
  # coerce all values to strings
  config = { k: str(v) for k, v in config.items() }
  # remove affixes from keys and values
  for k, v in config.copy().items():
    # values
    for p in value_prefixes:
      if v.startswith(p):
        config[k] = v[len(p):]
    for p in value_suffixes:
      if v.endswith(p):
        config[k] = v[:-len(p)]
    # keys
    for p in key_prefixes:
      if k.startswith(p):
        config[k[len(p):]] = config.pop(k)
    for p in key_suffixes:
      if k.endswith(p):
        config[k[:-len(p)]] = config.pop(k)
  # shorten keys to just first 3 characters in between each _
  config = { ''.join([x[:3] for x in k.split('_')]) : v for k, v in config.items() }
  # remove _ from values
  config = { k: v.replace('_', '') for k, v in config.items() }
  # make key, sorted alphabetically
  key = '_'.join([ f'{key}={value}' for key, value in sorted(config.items())])
  # strip all whitespace
  key = key.replace(' ', '')
  return key


if __name__ == '__main__':
  config = dict(
    dataset_cls='test',
    test_fn='test',
    num_classes=10,
    init_fn='test_init',
    none=None,
    dictionary={},
    ignoreme=Ignore('ignoreme'),
  )
  # TODO: figure out how to handle case where trimming to 3 chars results in a key collision
  print(make_key(config=config)) # Looks good! (for now)
  x = Ignore(2)
  print( x + 3 ) # 5 
  print( isinstance(x, Ignore) ) # True
  print( isinstance(x, int) ) # True
  ipdb.set_trace()
  x.abc # Should see: AttributeError: 'int' object has no attribute 'abc'