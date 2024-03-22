from collections import defaultdict
import cv2
import numpy as np
from ultralytics import trackers
from ultralytics import YOLO


def yolo_track(model, video_path):
    trackers.basetrack.BaseTrack.reset_id()
    results = model.track(source=video_path, 
                          conf=0.3, 
                          iou=0.5, 
                          show=False, 
                          verbose=False)
    return results


if __name__ == "__main__":
    model = YOLO("../../aicity2024_track5/weights/yolov8l-increase-augment-hr.pt")
    







