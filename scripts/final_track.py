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
#parser.add_argument('--video_id', type=str, required=False)
parser.add_argument('--ksplit', type=int, required=True)
parser.add_argument('-tyml', '--track_yaml_path', type=str, required=True)
# parser.add_argument('post_processing', type=str, required=True)
parser.add_argument('-pp1', '--post_processing1', action='store_true', help='Enable verbose output.')
parser.add_argument('-pp2', '--post_processing2', action='store_true', help='Enable verbose output.')

# Parse the arguments
args = parser.parse_args()

video_path = args.video_path
#video_id = args.video_id
ksplit = args.ksplit

# reset the tracker
trackers.basetrack.BaseTrack.reset_id()

# choose the split to work on
if ksplit == 1:
    video_ids = ['012', '014', '009', '019', '001', '002', '003', '004', '005', '006']
elif ksplit == 2:
    video_ids = ['017', '020', '022', '034', '007', '008', '010', '011', '013', '015']
elif ksplit == 3:
    video_ids = ['021', '023', '043', '047', '016', '018', '025', '026', '027', '029']
elif ksplit == 4: 
    video_ids = ['024', '028', '050', '053', '030', '031', '032', '033', '035', '036']
elif ksplit == 5:
    video_ids = ['040', '045', '068', '070', '037', '038', '039', '041', '042', '044']
elif ksplit == 6:
    video_ids = ['051', '055', '075', '095', '046', '048', '049', '052', '054', '056']
elif ksplit == 7:
    video_ids = ['061', '062', '085', '100', '058', '059', '060', '063', '064', '065']
elif ksplit == 8:
    video_ids = ['069', '072', '086', '067', '071', '073', '074', '076', '077', '078']
elif ksplit == 9:
    video_ids = ['081', '082', '087', '079', '080', '083', '084', '088', '089', '090']
elif ksplit == 10:   
    video_ids = ['093', '097', '092', '091', '094', '096', '098', '099', '057', '066']



# Load model
if args.model_type == "Y":
    model = YOLO(args.model_path)
if args.model_type == "T":
    model = RTDETR(args.model_path)


# Run inference
for video_id in video_ids:


    print(f"Processing: ksplit [{ksplit}] - video_id [{video_id}]")

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
                                        project=f'./results/ksplit{str(ksplit)}//{video_id}')


    iou_conditions_met = build_iou_array(results, num_frames = 200)

    all_groups = identify_group(iou_conditions_met)

    class_id_frame_map_array = class_id_frame_map(results, num_frames=200)

    max_frequency_groups, intersections_with_frequencies = modified_valid_group_identification(all_groups, appearance_threshold=0.2)

    missing_members = modified_find_missing_pair_member(max_frequency_groups, intersections_with_frequencies)

    frame_ids_with_valid_group_for_missing = find_frame_and_valid_group_for_missing_pairs(all_groups, missing_members, max_frequency_groups)

    closest_frames_for_missing_with_completing_elements = find_closest_frames_for_missing_id(all_groups, max_frequency_groups,frame_ids_with_valid_group_for_missing)

    class_id_frame_corrections = correct_class_id(class_id_frame_map_array, threshold=0.9)

    text_files_path = f'./results/ksplit{str(ksplit)}/{video_id}/track/labels'

    if args.post_processing1: 
        correct_post_1(text_files_path, video_id, class_id_frame_corrections)
    if args.post_processing2:
        line_insert_post_2(results, text_files_path, video_id, closest_frames_for_missing_with_completing_elements)