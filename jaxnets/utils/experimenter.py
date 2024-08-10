import numpy as np
from jax import Array
import jax
import jax.numpy as jnp
import jax.random as jr

from functools import partial

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from collections.abc import Generator

import equinox as eqx
from socket import gethostname
import os

from jaxnets.utils.admin import make_key

import ipdb

##########
## LOADING

def load_model(
  *, 
  save_dir : Callable = lambda: '../results' if gethostname() == 'Leons-MBP' else '/scratch/gpfs/ll4476/results',
  config_modifier : Callable | None = None, 
  **config
):
  # Create path key
  path_key = make_key(config=config)
  
  # Modify config if necessary
  if config_modifier is not None:
    config = config_modifier(config)
    
  seed = config['seed']
  model_cls = config['model_cls']
    
  # Get key (value should be irrelevant)
  model_key = jr.PRNGKey(0) # jr.split(jr.PRNGKey(seed), 4)[1]
  
  # Create empty model 
  model = model_cls(key=model_key, **config)
  
  # Get directory where results are saved
  savedir = save_dir()
  
  # Check for file
  if not os.path.exists(f'{savedir}/models/{path_key}.eqx'): # check if model exists
    raise FileNotFoundError(f'Model file not found: {path_key}.eqx')
  
  # Load model
  model = eqx.tree_deserialise_leaves(f'{savedir}/models/{path_key}.eqx', model)
  
  return model


def load(
  *,
  save_dir : Callable = lambda: '../results' if gethostname() == 'Leons-MBP' else '/scratch/gpfs/ll4476/results',
  **config
):
  """
  Load metrics, model, and weights for a given config.
  """
  # Get directory where results are saved
  savedir = save_dir()
  # Make sure directories exist
  os.makedirs(savedir, exist_ok=True)
  os.makedirs(f'{savedir}/metrics', exist_ok=True)
  os.makedirs(f'{savedir}/models', exist_ok=True)
  os.makedirs(f'{savedir}/weights', exist_ok=True)
  # Create path key
  path_key = make_key(config=config)
  want_model = config.get('save_model', False)
  want_weights = config.get('save_weights', False)
  # Check if metrics, model, and weights are saved
  if path_key + '.npz' not in os.listdir(f'{savedir}/metrics'):
    raise FileNotFoundError('Results for ' + path_key + ' not found') # Metrics are always saved
  metrics = jnp.load(f'{savedir}/metrics/' + path_key + '.npz', allow_pickle=True)['metrics']
  have_model = (path_key + '.eqx' in os.listdir(f'{savedir}/models'))
  have_weights = (path_key + '.npz' in os.listdir(f'{savedir}/weights'))
  
  # Load model and weights, if requested
  if want_model:
    if have_model:
      config_modifier = config.pop('config_modifier', None)
      model = load_model(config_modifier=config_modifier, **config)
    else:
      raise FileNotFoundError('Model for ' + path_key + ' not found')
    
  if want_weights:
    if have_weights:
      weights = jnp.load(f'{savedir}/weights/' + path_key + '.npz', allow_pickle=True)
      weights = list(weights.values())
    else:
      raise FileNotFoundError('Weights for ' + path_key + ' not found')
    
  # Return metrics, model, and weights as requested, if available
  if want_model and want_weights:
    print('Found metrics, model, and weights for ' + path_key)
    return metrics, weights, model
  if want_model:
    print('Found metrics and model for ' + path_key)
    return metrics, model
  if want_weights:
    print('Found metrics and weights for ' + path_key)
    return metrics, weights
  print('Found metrics for ' + path_key)
  return metrics


#############
## SIMULATING

def simulate(**config):
  if 'task' not in config:
    raise ValueError('No task specified')
  simulate_fn = config['task']
  return simulate_fn(**config)

def simulate_or_load(**config):
  try:
    return load(**config)
  except FileNotFoundError as e:
    print(e)
    print('Simulating')
    return simulate(**config)
  
###########
## BATCHING

def batcher(
  sampler: Sequence, 
  batch_size: int,
  length: int | None = None) -> Generator[Sequence, None, None]:
  """Batch a sequence of examples."""
  n = length or len(sampler)
  for i in range(0, n, batch_size):
    yield sampler[i : min(i + batch_size, n)]
  
  
####################
## WEIGHT EXTRACTION

def is_linear(x):
  return isinstance(x, eqx.nn.Linear)

def is_array(x):
  return eqx.is_array(x)

def get_linears(m):
  return [x.weight for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(x)]

def get_weights(m):
  w = []
  for x in jax.tree_util.tree_leaves(m, is_leaf=is_array):
    if is_linear(x):
      w.append(x.weight)
    elif is_array(x):
      w.append(x)
  return w

def repack_weights(weights):
  # get shapes from first entry
  shapes = [w.shape for w in weights[0]]
  # create list of weight arrays
  w = [np.zeros((len(weights), *s)) for s in shapes]
  # fill in the arrays
  for t, weight in enumerate(weights):
    for i, w_ in enumerate(weight):
      w[i][t] = w_
  return w