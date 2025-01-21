#!/bin/bash

# device: 0

DATASET="sdogs"
EXP="lambda"
MODE="real"
CONFIG_DIR="./configs/"$MODE"/"$DATASET"/"$EXP""
PYTHON_SCRIPT="./real_main.py"

for config_file in "$CONFIG_DIR"/*.yml; do
  if [ -e "$config_file" ]; then
    echo "Running configuration: $config_file"
    python "$PYTHON_SCRIPT" --config "$config_file"
  else
    echo "No configuration files found in $CONFIG_DIR"
    break
  fi
done
