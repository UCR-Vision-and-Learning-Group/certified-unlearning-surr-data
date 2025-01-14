#!/bin/bash

# Directory containing the configuration files
CONFIG_DIR="./certified-unlearning-surr-data/configs/real/cifar10/fratio"

# Python script to run with each configuration file
PYTHON_SCRIPT="./certified-unlearning-surr-data/real_main.py"

# device: 2

# Loop through each configuration file in the directory
for config_file in "$CONFIG_DIR"/*.yml; do
  echo "Running configuration: $config_file"
  python "$PYTHON_SCRIPT" --config "$config_file"
done
