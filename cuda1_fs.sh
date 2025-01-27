#!/bin/bash

# device: 2

DATASET="caltech256"
EXP="dirichlet-fs"
MODE="real"
CONFIG_DIR="./results/"$MODE"/"$DATASET"/"$EXP""
PYTHON_SCRIPT="./real_forget_score.py"

for config_dir in "$CONFIG_DIR"/*/; do
  if [ -d "$config_dir" ]; then
    echo "Running configuration directory: $config_dir"
    python "$PYTHON_SCRIPT" --base "$config_dir" --device "1"
  else
    echo "No configuration directories found in $CONFIG_DIR"
    break
  fi
done