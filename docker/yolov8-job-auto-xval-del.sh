#!/bin/bash
num_class=9
split_start=1
split_end=10
split_increment=1

for (( split=split_start; split<=split_end; split+=split_increment )); do
    #video_id=$(printf "%03d" $split)
    #job_name="yolo-job-jacket-increase-augment-v8-${num_class}class-xval-${split}"
    job_name="rtdetr-job-jacket-increase-augment-${num_class}class-xval-${split}"
    kubectl delete job $job_name
done