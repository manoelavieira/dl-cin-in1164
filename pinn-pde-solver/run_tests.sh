#!/bin/bash

# Define an array of configuration files
CONFIG_FILES=('conv-adaptive-v2.json')

# Iterate over the array and run the python script with the specified arguments
for file in "${CONFIG_FILES[@]}"
do
    python3 main.py --config_file "$file" --num_seeds 0 --save_outputs True --visualize True --no_wandb True
done
