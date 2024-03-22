import numpy as np
import pandas as pd
import os
import sys

if __name__ == '__main__':

    path = "results_yolov8l_increase_augment_hr_iou_0.5/"

    combined = []
    for i in range(1,101):
        df = pd.read_csv(path+"submission_{}.txt".format(i))
        combined.append(df)
    
    combined = pd.concat(combined)
     
    combined["bb_left"] = combined["bb_left"] - combined["bb_width"]/2
    combined["bb_top"] = combined["bb_top"] - combined["bb_height"]/2

    combined["bb_left"] = combined["bb_left"].clip(lower=0, upper=1920)
    combined["bb_top"] = combined["bb_top"].clip(lower=0, upper=1080)
    combined["bb_width"] = combined["bb_width"].clip(lower=1, upper=1920)
    combined["bb_height"] = combined["bb_height"].clip(lower=1, upper=1080)

    combined["class"] = combined["class"] + 1

    combined.to_csv(path+"vigir_submission_7.txt", 
                    index=False, 
                    header=None)

