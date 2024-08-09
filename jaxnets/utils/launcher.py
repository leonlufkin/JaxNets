"""
Launching utilities
timestamping, executor setup, etc.

Author: Leon Lufkin
Last modified: 2024-08-08
"""

import os
import datetime
from pathlib import Path
from submit import get_submitit_executor, submit_jobs, product_kwargs
from tqdm import tqdm

def get_timestamp():
  """
  Return a date and time `str` timestamp.
  Format: YYYY-MM-DD_HH-MM-SS
  """
  return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def get_executor(
  job_name: str, 
  cluster: str = "slurm", 
  partition: str = "cpu", 
  timeout_min: int = 60, 
  mem_gb: int = 10, 
  parallelism: int = 30, 
  gpus_per_node: int = 0,
  log_base: str = "",
  debug_base: str = ""):
  """
  Parameters
  ----------
  job_name (str): Name of the job.
  cluster (str): "slurm" or "debug".
  partition (str): "cpu" or "gpu".
  parallelism (int): Max number of jobs to run in parallel.
  gpus_per_node (int): Number of GPUs per node.
  """
  
  executor = get_submitit_executor(
    timeout_min=timeout_min,
    mem_gb=mem_gb,
    # NOTE: `log_dir` should be set to a directory shared across the head
    # (launching) node as well as compute nodes;
    # can set `export RESULTS_HOME="..." external to Python or
    # change the below.
    log_dir=Path(
        os.path.join(os.environ.get("LOGS_HOME"), job_name)
        if os.environ.get("LOGS_HOME") is not None
        else os.path.join(log_base, os.environ.get("USER"), "logs", job_name),
        get_timestamp(),
    ) if cluster != "debug" else Path(debug_base, job_name, get_timestamp()),
    # NOTE: Use `cluster="debug"` to simulate a SLURM launch locally.
    cluster=cluster,
    # NOTE: This may be specific to your cluster configuration.
    # Run `sinfo -s` to get partition information.
    slurm_partition=partition,
    slurm_parallelism=parallelism,
    gpus_per_node=gpus_per_node,
  )
  
  return executor