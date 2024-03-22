from collections import defaultdict
import cv2
import numpy as np
from ultralytics import trackers
from ultralytics import YOLO


def convert_bbox_to_corners(bbox):
    cx, cy, w, h = bbox
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return [x1, y1, x2, y2]


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def iou_condition_met(track_id_i, track_id_j, frame):
    i_track_id = track_id_i[0]
    i_box = track_id_i[1]
    i_class = track_id_i[2]

    j_track_id = track_id_j[0]
    j_box = track_id_j[1]
    j_class = track_id_j[2]

    score = iou(convert_bbox_to_corners(i_box),convert_bbox_to_corners(j_box))

    if score > 0.20:
        return True 
    else:
        return False


def yolo_track(model, video_path):

    trackers.basetrack.BaseTrack.reset_id()
    results = model.track(source=video_path, conf=0.3, iou=0.5, show=False, verbose=False)

    return results


def build_iou_array(results, num_frames = 200):

    max_tracks = trackers.basetrack.BaseTrack.next_id() - 1 # This should be set to the maximum number of track IDs
    # max_tracks = 31

    # # Initialize the 3D array
    iou_conditions_met = np.zeros((num_frames, max_tracks+1, max_tracks+1), dtype=int)
    track_ids_per_frame = {}
    # # Example track IDs per frame (replace with your actual method of obtaining them)
    for frame in range(num_frames):
        if results[frame].boxes.id is not None:
            track_ids_per_frame.update({frame: results[frame].boxes.id.int().cpu().tolist()})
        else:
            track_ids_per_frame.update({frame: []})
            
    # # Populate the array
    for frame in range(num_frames):
        frame_classes = results[frame].boxes.cls.cpu().tolist()
        frame_boxes = results[frame].boxes.xywh.cpu().tolist()
        track_ids = track_ids_per_frame[frame]
        for i in range(len(track_ids)):
            for j in range(len(track_ids)):
                if i != j and iou_condition_met([track_ids[i], frame_boxes[i], frame_classes[i]], [track_ids[j], frame_boxes[j], frame_classes[j]], frame):
                    iou_conditions_met[frame, track_ids[i], track_ids[j]] = 1
    
    return iou_conditions_met


def identify_group(built_array):
    all_groups = {}
    for frame in range(built_array.shape[0]):

        test_frame = built_array[frame].squeeze()

        groups = []
        
        ids = test_frame
        r, c = np.shape(ids)
        for i in range(r):
        idx = np.where(ids[i, :] == 1)[0]
        idx = list(idx)
        if len(idx) > 0:
            if set(idx) in groups:
            continue
            else:
            groups.append(set(idx))
        # print("Frame = ", frame," groups = ", groups)
        all_groups.update({frame: groups})
    
    return all_groups
