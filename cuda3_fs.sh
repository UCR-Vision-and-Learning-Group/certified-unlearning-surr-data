#!/bin/bash

# device: 3

DATASET="cifar10"
EXP="dirichlet/conv2"
MODE="real"
CONFIG_DIR="./results/"$MODE"/"$DATASET"/"$EXP""
PYTHON_SCRIPT="./real_forget_score.py"

for config_dir in "$CONFIG_DIR"/*/; do
  if [ -d "$config_dir" ]; then
    echo "Running configuration directory: $config_dir"
    python "$PYTHON_SCRIPT" --base "$config_dir" --device "3"
  else
    echo "No configuration directories found in $CONFIG_DIR"
    break
  fi
done