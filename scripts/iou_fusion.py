import numpy as np
import pandas as pd


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

def convert_bbox_to_corners(bbox):
        cx, cy, w, h = bbox
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return [x1, y1, x2, y2]

def read_prediction(file_path):
    
    fdf = pd.read_csv(file_path, sep=' ', header=None, names=["class", "bb_left", "bb_top", "bb_width", "bb_height", "confidence"])
    fdf["bb_top"] = fdf["bb_top"]*1080 
    fdf["bb_width"] = fdf["bb_width"]*1920 
    fdf["bb_height"] = fdf["bb_height"]*1080 
    fdf["bb_left"] = fdf["bb_left"] - fdf["bb_width"]/2
    fdf["bb_top"] = fdf["bb_top"] - fdf["bb_height"]/2
    
    classes_array = fdf['class'].to_numpy()
    Bounding_boxes_array = fdf[['bb_left', 'bb_top', 'bb_width', 'bb_height']].to_numpy()    
    
    return [classes_array, Bounding_boxes_array]


def simple_fuse_prediction(two_class_pred_path, three_class_pred_path, five_class_pred_path, nine_class_pred_path, alpha=0.5):

    [two_classes_array, two_Bounding_boxes_array] = read_prediction(two_class_pred_path)
    [three_classes_array, three_Bounding_boxes_array] = read_prediction(three_class_pred_path)
    [five_classes_array, five_Bounding_boxes_array] = read_prediction(five_class_pred_path)
    [nine_classes_array, nine_Bounding_boxes_array] = read_prediction(nine_class_pred_path)

    print(two_classes_array.shape)
    print(three_classes_array.shape)
    print(five_classes_array.shape)

    final_classes = []
    final_boxes = []

    for i, nine_box in enumerate(nine_Bounding_boxes_array):
        nine_box_corners = convert_bbox_to_corners(nine_box)
        add_nine_class_detection = False

        #appending the 9-class predictions
        final_classes.append(nine_classes_array[i])
        final_boxes.append(nine_box)

        all_other_boxes = np.vstack((two_Bounding_boxes_array, three_Bounding_boxes_array, five_Bounding_boxes_array))
        all_other_classes = np.concatenate((two_classes_array, three_classes_array, five_classes_array))

        #Checking the IoU for all the other predictions with the 9-class prediction and 
        for other_class, other_box in zip(all_other_classes, all_other_boxes):
            other_box_corners = convert_bbox_to_corners(other_box)
            if iou(nine_box_corners, other_box_corners) > alpha:
                #Changing the class for the other predictions to the class from 9-class
                #prediction and appending the boxes to the final predictions
                final_classes.append(nine_classes_array[i])
                final_boxes.append(other_box)
            

    return final_classes, final_boxes




if __name__ == "__main__":

    two_class_pred_path = "results_yolov8l_increase_augment_2class/1/labels/00000035.txt"
    three_class_pred_path = "results_yolov8l_increase_augment_3class/1/labels/00000035.txt"
    five_class_pred_path = "results_yolov8l_increase_augment_5class/1/labels/00000035.txt"
    nine_class_pred_path = "results_yolov8l_increase_augment/1/labels/00000035.txt"

    final_classes, final_boxes = simple_fuse_prediction(two_class_pred_path, three_class_pred_path, five_class_pred_path, nine_class_pred_path, alpha=0.5)
    print(len(final_classes))
    print(len(final_boxes))





