from collections import defaultdict
from collections import Counter
import cv2
import numpy as np
from ultralytics import trackers
from ultralytics import YOLO
import torch
from tracking_script import *
import argparse
from ultralytics import RTDETR


parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--model_type', type=str, required=True)
parser.add_argument('-m', '--model_path', type=str, required=True)
parser.add_argument('--video_path', type=str, required=True)
parser.add_argument('--video_id', type=str, required=True)
parser.add_argument('-tyml', '--track_yaml_path', type=str, required=True)
# parser.add_argument('post_processing', type=str, required=True)
parser.add_argument('-pp1', '--post_processing1', action='store_true', help='Enable verbose output.')
parser.add_argument('-pp2', '--post_processing2', action='store_true', help='Enable verbose output.')

# Parse the arguments
args = parser.parse_args()

if args.model_type == "Y":
    model = YOLO(args.model_path)
if args.model_type == "T":
    model = RTDETR(args.model_path)
video_path = args.video_path
video_id = args.video_id

trackers.basetrack.BaseTrack.reset_id()

results = model.track(source=f"{video_path}/{video_id}.mp4", 
                                    tracker=args.track_yaml_path, 
                                    conf=0.1, 
                                    iou=0.1, 
                                    show=False, 
                                    save_conf = True, 
                                    save_txt=True, 
                                    verbose=False, 
                                    save_frames=True, 
                                    save=True, 
                                    project=f'./results+{args.video_id}')


iou_conditions_met = build_iou_array(results, num_frames = 200)

all_groups = identify_group(iou_conditions_met)

class_id_frame_map_array = class_id_frame_map(results, num_frames=200)

max_frequency_groups, intersections_with_frequencies = valid_group_identification(all_groups)

missing_members = find_missing_pair_member(max_frequency_groups, intersections_with_frequencies)

frame_ids_with_valid_group_for_missing = find_frame_and_valid_group_for_missing_pairs(all_groups, missing_members, max_frequency_groups)

closest_frames_for_missing_with_completing_elements = find_closest_frames_for_missing_id(all_groups, max_frequency_groups,frame_ids_with_valid_group_for_missing)

class_id_frame_corrections = correct_class_id(class_id_frame_map_array, threshold=0.9)

text_files_path = f'./results+{args.video_id}/track/labels'

if args.post_processing1: 
    correct_post_1(text_files_path, video_id, class_id_frame_corrections)
if args.post_processing2:
    line_insert_post_2(results, text_files_path, video_id, closest_frames_for_missing_with_completing_elements)