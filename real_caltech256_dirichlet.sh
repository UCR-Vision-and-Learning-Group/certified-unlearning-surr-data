#!/bin/bash

# Directory containing the configuration files
CONFIG_DIR="/home/umityigitbsrn/Desktop/umityigitbsrn/certified-unlearning-surr-data/configs/real/caltech256/dirichlet"

# Python script to run with each configuration file
PYTHON_SCRIPT="/home/umityigitbsrn/Desktop/umityigitbsrn/certified-unlearning-surr-data/real_main.py"

# device: 3

# Loop through each configuration file in the directory
for config_file in "$CONFIG_DIR"/*.yml; do
  echo "Running configuration: $config_file"
  python "$PYTHON_SCRIPT" --config "$config_file"
done
