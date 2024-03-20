#!/bin/bash
num_class=3
split_start=1
split_end=1
split_increment=1

for (( split=split_start; split<=split_end; split+=split_increment )); do
    job_name="yolo-job-jacket-increase-augment-v8-${num_class}class-xval-${split}"
    kubectl delete job $job_name
done