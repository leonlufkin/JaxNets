import os
import subprocess
import sys
import argparse

def find_project_details():
  current_path = os.getcwd()
  path_components = current_path.split(os.path.sep)
  
  if len(path_components) >= 2:
    dir_name = path_components[-1]
    project_name = path_components[-2]
    return project_name, dir_name
  else:
    raise ValueError("Current directory structure does not match .../PROJECT_NAME/DIR_NAME")

def run_pull_recent_hpc(project_name, dir_name, time_min):
  script_path = "/Users/leonlufkin/Documents/Github/JaxNets/jaxnets/utils/pull_recent_hpc.sh"
  
  if not os.path.exists(script_path):
    raise FileNotFoundError(f"The script {script_path} was not found in the current directory.")
  
  try:
    subprocess.run(["bash", script_path, project_name, dir_name, str(time_min)], check=True)
    print(f"Successfully ran pull_recent_hpc.sh for {project_name}/{dir_name}")
  except subprocess.CalledProcessError as e:
    print(f"Error running pull_recent_hpc.sh: {e}")
    sys.exit(1)

def pull_recent_hpc(time=60):
  try:
    project_name, dir_name = find_project_details()
    # print(f"Detected PROJECT_NAME: {project_name}")
    # print(f"Detected DIR_NAME: {dir_name}")
    # print(f"Using TIME: {time} minutes")
    
    run_pull_recent_hpc(project_name, dir_name, time)
  except ValueError as e:
    print(f"Error: {e}")
    sys.exit(1)
  except FileNotFoundError as e:
    print(f"Error: {e}")
    sys.exit(1)

def parse_arguments():
  parser = argparse.ArgumentParser(description="Run pull_recent_hpc.sh with detected project details.")
  parser.add_argument("time", nargs="?", type=int, default=60, help="Time in minutes for recent files (default: 60)")
  parser.add_argument("--time", type=int, dest="time_opt", help="Time in minutes for recent files (optional named argument)")
  return parser.parse_args()

if __name__ == "__main__":
  args = parse_arguments()
  time = args.time_opt if args.time_opt is not None else args.time
  pull_recent_hpc(time)