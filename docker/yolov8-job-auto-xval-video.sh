#!/bin/bash
#num_class=9
split_start=3
split_end=10
split_increment=1

for (( split=split_start; split<=split_end; split+=split_increment )); do
    #export num_class="$num_class"
    export split="$split"
    #export video_id=$(printf "%03d" $split)
    #envsubst < job-template-gpu.yaml | kubectl apply -f -
    envsubst < yolov8-job-auto-xval-video.yaml | kubectl apply -f -
done