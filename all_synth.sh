#!/bin/bash

# Base directory containing the synthetic folder
BASE_DIR="./configs/synthetic"

# Loop through all folders under synthetic
for folder in "$BASE_DIR"/*; do
    if [ -d "$folder" ]; then
        # Loop through all YAML files in the current folder
        for config_file in "$folder"/*.yml; do
            if [ -f "$config_file" ]; then
                echo "Running config: $config_file"
                python synthetic_main.py --config "$config_file"
            fi
        done
    fi
done
