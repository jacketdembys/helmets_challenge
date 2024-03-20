#!/bin/bash
num_class=2
split_start=1
split_end=10
split_increment=1

for (( split=split_start; split<=split_end; split+=split_increment )); do
    job_name="yolo-job-jacket-increase-augment-v8-${num_class}class-xval"
    kubectl delete job $job_name
done