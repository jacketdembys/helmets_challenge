#!/bin/bash

# Define the Python script path
python_script="detect_yolov8.py"

# Define the start and end indexes
start_index=1
end_index=100
increment=5

# Loop through start and end indexes in steps of increment
for ((i=$start_index; i<=$end_index; i+=increment)); do
    # Calculate the end index for this iteration
    end=$((i + increment - 1))
    if [ "$end" -gt "$end_index" ]; then
        end=$end_index
    fi

    # Run the Python script with the current start and end indexes
    python $python_script -sidx $i -eidx $end &
done

# Wait for all background processes to finish
wait

echo "All processes have completed."
