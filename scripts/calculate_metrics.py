import numpy as np
import pandas as pd
import time

import numpy as np
import csv
import argparse



# compute epoch time
def compute_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_hours = int(elapsed_time / 3600)  # Calculate elapsed hours
    elapsed_time -= elapsed_hours * 3600  # Subtract elapsed hours
    elapsed_mins = int(elapsed_time / 60)  # Calculate elapsed minutes
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))  # Calculate elapsed seconds
    return elapsed_hours, elapsed_mins, elapsed_secs  # Return elapsed hours, minutes, and seconds



def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    Args:
        box1 (list): [x1, y1, w1, h1] representing the coordinates of the first bounding box.
        box2 (list): [x1, y1, w1, h1] representing the coordinates of the second bounding box.
    Returns:
        float: IoU value.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection area
    x_intersection = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_intersection = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    intersection = x_intersection * y_intersection
    
    # Calculate union area
    union = w1 * h1 + w2 * h2 - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    
    return iou



def compute_precision_recall_old(gt_boxes, pred_boxes, threshold=0.1):
    """
    Compute precision and recall given ground truth and predicted bounding boxes.
    Args:
        gt_boxes (list): List of ground truth bounding boxes.
        pred_boxes (list): List of predicted bounding boxes.
        threshold (float): IoU threshold for matching.
    Returns:
        float: Precision
        float: Recall
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for pred_box in pred_boxes:
        max_iou = 0
        for gt_box in gt_boxes:
            iou = compute_iou(pred_box, gt_box)
            if iou > max_iou:
                max_iou = iou
        if max_iou >= threshold:
            true_positives += 1
        else:
            false_positives += 1
            
    print("True Positives:", true_positives)
    false_negatives = len(gt_boxes) - true_positives

    print("False Negatives:", false_negatives)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    print("Recall:", recall)

    return precision, recall


def compute_precision_recall(gt_boxes, pred_boxes, threshold=0.1):
    """
    Compute precision and recall given ground truth and predicted bounding boxes.
    Args:
        gt_boxes (list): List of ground truth bounding boxes.
        pred_boxes (list): List of predicted bounding boxes.
        threshold (float): IoU threshold for matching.
    Returns:
        float: Precision
        float: Recall
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Track which ground truth bounding boxes have been matched
    matched_gt = set()
    
    # Iterate over each predicted bounding box
    for pred_box in pred_boxes:
        max_iou = 0
        for i, gt_box in enumerate(gt_boxes):
            iou = compute_iou(pred_box, gt_box)
            if iou > max_iou:
                max_iou = iou
                max_iou_idx = i
        if max_iou >= threshold:
            # If IoU is above the threshold, consider it a true positive
            true_positives += 1
            # Mark the matched ground truth bounding box
            matched_gt.add(max_iou_idx)
        else:
            # Otherwise, count it as a false positive
            false_positives += 1
    
    # Count the unmatched ground truth bounding boxes as false negatives
    false_negatives = len(gt_boxes) - len(matched_gt)
    
    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return precision, recall




def compute_f1_score(precision, recall):
    """
    Compute F1 score given precision and recall.
    Args:
        precision (float): Precision value.
        recall (float): Recall value.
    Returns:
        float: F1 score.
    """
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score

def compute_map(gt_boxes, pred_boxes, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    """
    Compute Mean Average Precision (MAP) at different IoU thresholds.
    Args:
        gt_boxes (list): List of ground truth bounding boxes.
        pred_boxes (list): List of predicted bounding boxes.
        iou_thresholds (numpy.ndarray): Array of IoU thresholds.
    Returns:
        float: MAP50-95
        float: MAP50
    """
    precision_recall_values = []
    
    for threshold in iou_thresholds:
        precision, recall = compute_precision_recall(gt_boxes, pred_boxes, threshold)
        precision_recall_values.append((precision, recall))
    
    # Compute MAP50-95
    ap_values = [precision * recall for precision, recall in precision_recall_values]
    map50_95 = np.mean(ap_values)
    
    # Compute MAP50
    map50_index = np.where(iou_thresholds == 0.5)[0][0]
    map50 = ap_values[map50_index]
    
    return map50_95, map50

# Read bounding box coordinates from file
def read_bounding_boxes_from_file(filename):
    """
    Read bounding box coordinates from a file.
    Args:
        filename (str): Name of the file containing bounding box information.
    Returns:
        dict: A dictionary where keys are video IDs and values are lists of bounding box coordinates.
    """
    bounding_boxes = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            video_id = int(row[0])
            box = list(map(int, row[2:6]))
            if video_id not in bounding_boxes:
                bounding_boxes[video_id] = []
            bounding_boxes[video_id].append(box)
    return bounding_boxes


if __name__ == "__main__":

    # Example usage:
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Finding the aggregated cross-validation results")
    parser.add_argument('-gt', type=str, help='groung truth paths', default='/home/retina/dembysj/gt/')
    parser.add_argument('-pred', type=str, help='prediction paths', default='results_wo_pp/')
    args = parser.parse_args()


    # Define the filenames for the files containing bounding box coordinates
    #gt_filename = "/home/retina/dembysj/Dropbox/WCCI2024/challenges/aicity2024_track5/aicity2024_track5_train/gt.txt"
    #pred_filename = "/home/retina/dembysj/Dropbox/WCCI2024/challenges/aicity2024_track5/aicity2024_track5_train/gt.txt"

    #ground_truth_boxes = pd.read_csv("/home/retina/dembysj/Dropbox/WCCI2024/challenges/aicity2024_track5/aicity2024_track5_train/gt.txt", header=None, sep=",")
    #ground_truth_boxes = np.array(ground_truth_boxes)  #[:,2:6]
    #yolo_pred_boxes = ground_truth_boxes.copy()

    xval_all_precision = []
    xval_all_recall = []
    xval_all_f1_score = []
    xval_all_map = []
    xval_all_map_95 = []

    for i in range(1,11,1):

        # Define the filenames for the files containing bounding box coordinates
        gt_filename = f'{args.gt}ksplit{i}/combined_ksplit{i}.txt'
        pred_filename = f'{args.pred}ksplit{i}/combined_ksplit{i}.txt'

        # Read ground truth and predicted bounding box coordinates from files
        ground_truth_boxes = read_bounding_boxes_from_file(gt_filename)
        predicted_boxes = read_bounding_boxes_from_file(pred_filename)


        all_precision = []
        all_recall = []
        all_f1_score = []
        all_map = []
        all_map_95 = []

        # Iterate over each video ID present in the ground truth data
        for video_id, gt_boxes in ground_truth_boxes.items():
            # Check if there are corresponding predicted bounding boxes for this video ID
            if video_id in predicted_boxes:
                # Retrieve predicted bounding boxes for this video ID
                pred_boxes = predicted_boxes[video_id]

                # Compute evaluation metrics for the current video ID
                map50_95, map50 = compute_map(gt_boxes, pred_boxes)
                precision, recall = compute_precision_recall(gt_boxes, pred_boxes)
                f1_score = compute_f1_score(precision, recall)

                # Print the evaluation metrics for the current video ID
                print()    
                print(f"Metrics for Ksplit {i} - Video ID {video_id}:")
                print("MAP50-95:", map50_95)
                print("MAP50:", map50)
                print("Precision:", precision)
                print("Recall:", recall)
                print("F1 Score:", f1_score)
                print()

                # Store evaluation metrics for this video ID
                all_precision.append(precision)
                all_recall.append(recall)
                all_f1_score.append(f1_score)
                all_map.append(map50)
                all_map_95.append(map50_95)

            else:
                print(f"No predicted bounding boxes found for Video ID {video_id}")


        # Compute aggregated evaluation metrics over all video IDs
        overall_precision = np.mean(all_precision)
        overall_recall = np.mean(all_recall)
        overall_f1_score = np.mean(all_f1_score)
        overall_map = np.mean(all_map)
        overall_map_95 = np.mean(all_map_95)

        # Print the aggregated evaluation metrics
        print()
        print(f"Overall Metrics for Ksplit {i}:")
        print("ksplit {} overall Precision: {}".format(i, overall_precision))
        print("ksplit {} overall Recall: {}".format(i, overall_recall))
        print("ksplit {} overall F1 Score: {}".format(i, overall_f1_score))
        print("ksplit {} overall mAP50: {}".format(i, overall_map))
        print("ksplit {} overall mAP50_95: {}".format(i, overall_map_95))

        # Store evaluation metrics for this video ID
        xval_all_precision.append(overall_precision)
        xval_all_recall.append(overall_recall)
        xval_all_f1_score.append(overall_f1_score)
        xval_all_map.append(overall_map)
        xval_all_map_95.append(overall_map_95)


    # Compute aggregated evaluation metrics over all video IDs
    overall_xval_precision = np.mean(xval_all_precision)
    overall_xval_recall = np.mean(xval_all_recall)
    overall_xval_f1_score = np.mean(xval_all_f1_score)
    overall_xval_map = np.mean(xval_all_map)
    overall_xval_map_95 = np.mean(xval_all_map_95)

    # Print the aggregated evaluation metrics
    print()
    print(f"Xval Overall Metrics:")
    print("Xval overall Precision: {}".format(overall_xval_precision))
    print("Xval overall Recall: {}".format(overall_xval_recall))
    print("Xval overall F1 Score: {}".format(overall_xval_f1_score))
    print("Xval overall mAP50: {}".format(overall_xval_map))
    print("Xval overall mAP50_95: {}".format(overall_xval_map_95))


    end_time = time.time()

    hours, mins, secs = compute_time(start_time, end_time)
    print(f"Elapsed time: {hours} hours, {mins} minutes, {secs} seconds")
