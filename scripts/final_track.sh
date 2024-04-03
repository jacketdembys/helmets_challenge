#!/bin/bash

# Define the Python script path
python_script="final_track.py"

# Define the start and end indexes
start_index=1
end_index=10
increment=1

# Loop through start and end indexes in steps of increment
for ((i=$start_index; i<=$end_index; i+=increment)); do

    # Run the Python script with the current start and end indexes
    python $python_script -m ../../aicity2024_track5/weights/yolov8-video-xavl-9class-split${i}.pt --model_type Y --video_path ../../aicity2024_track5/aicity2024_track5_train/videos  -tyml ./botsort.yaml --ksplit ${i} -pp2 & # -pp1 -pp2 &

done

# Wait for all background processes to finish
wait

echo "All processes have completed."