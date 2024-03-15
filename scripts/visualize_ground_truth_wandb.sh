#!/bin/bash

# Define the Python script path
python_script="visualize_ground_truth_wandb.py"

# Define the base video ID
base_videoid="001"

# Loop through 100 video IDs
for ((i=71; i<=80; i++)); do
    # Generate the current video ID with leading zeros
    videoid=$(printf "%03d" $i)

    # Print the current video ID
    echo "Processing video ID: $videoid"

    # Run the Python script with the current video ID
    python $python_script -videoid "$videoid" &
done

# Wait for all background processes to finish
wait

echo "All processes have completed."
