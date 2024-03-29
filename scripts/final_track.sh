#!/bin/bash

# Define the Python script path
python_script="final_track.py"

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
    vids=$(printf "%03d" $i)
    vide=$(printf "%03d" $end)
    python $python_script -m ../../aicity2024_track5/weights/yolov8l-increase-augment-all.pt --model_type Y --video_path ../../aicity2024_track5/aicity2024_track5_test/videos  --video_id $vids --video_ide $vide -tyml ./botsort.yaml -pp1 -pp2 &
done

# Wait for all background processes to finish
wait

echo "All processes have completed."