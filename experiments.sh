#!/usr/bin/env bash

set -e

# Check if WANDB_API_KEY is set
if [ -z "${WANDB_API_KEY}" ]; then
  echo "Error: WANDB_API_KEY environment variable is not set"
  exit 1
fi

./monty_experiment.sh run_parallel -e base_77obj_dist_agent
./monty_experiment.sh run_parallel -e base_77obj_surf_agent
./monty_experiment.sh run_parallel -e randrot_noise_77obj_dist_agent
./monty_experiment.sh run_parallel -e randrot_noise_77obj_surf_agent
