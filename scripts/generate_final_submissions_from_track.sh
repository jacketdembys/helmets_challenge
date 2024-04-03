#!/bin/bash

# set the iteration index
start_index=2
end_index=10
increment=1

# set path to folder
path_to_folder="/home/retina/dembysj/gt"


# Loop through ksplit1 to ksplit10
for ((i=start_index; i<=end_index; i+=increment)); do
    ksplit="ksplit$i"
    python generate_final_submissions_from_track.py -path "$path_to_folder" -ksplit "$ksplit" &
done


# Wait for all background processes to finish
wait

echo "All processes have completed."
