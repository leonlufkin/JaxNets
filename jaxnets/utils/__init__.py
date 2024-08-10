"""`Sampler`s operating over `Dataset`s."""
from jaxnets.utils.admin import Ignore, make_key
# Import these as needed from launcher.py, will break on a local machine
# from jaxnets.utils.launcher import 
from jaxnets.utils.experimenter import load_model, load, simulate, simulate_or_load, is_linear, is_array, get_linears, get_weights, repack_weights, batcher
from jaxnets.utils.metrics import accuracy, mse, ce
from jaxnets.utils.pull_recent_hpc import pull_recent_hpc
from jaxnets.utils.sweeper import tupify, sweep_func
from jaxnets.utils.submit import submit_jobs, product_kwargs
# from jaxnets.utils.visualization import 

__all__ = (
  # admin.py
  "Ignore", "make_key",
  # launcher.py 
  "tupify", "sweep_func",
  # experimenter.py
  "load_model", "load", "simulate", "simulate_or_load", "is_linear", "is_array", "get_linears", "get_weights", "repack_weights", "batcher",
  # metrics.py
  "accuracy", "mse", "ce",
  # pull_recent_hpc.py
  "pull_recent_hpc",
  # submit.py
  "submit_jobs", "product_kwargs",
)
