import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

if __name__ == '__main__':

    path = "results_final/"
    predicted_labels = Path(path)
    predicted_labels = sorted(predicted_labels.rglob("*.txt"))
    #print(paths)

    combined = []
    for i in range(len(predicted_labels)):
        filename = predicted_labels[i].stem
        filename = filename.split("_")
        videoid = int(filename[0])
        frameid = int(filename[1])

        print("Prediction ... Processing video id [{}] - frame id [{}]".format(videoid, frameid))

        df = pd.read_csv(predicted_labels[i], header=None, sep=" ")
        if len(df.columns) == 6:
            df.columns = ["class", "bb_left", "bb_top","bb_width", "bb_height", "confidence"]
        elif len(df.columns) == 7: 
            df.columns = ["class", "bb_left", "bb_top","bb_width", "bb_height", "confidence", "track_id"]

        for indx in df.index:
            if df["bb_left"][indx] == np.nan or df["bb_left"][indx] == np.Inf or df["bb_left"][indx] == -np.inf  or df["bb_top"][indx] == np.nan or df["bb_top"][indx] == np.inf or df["bb_left"][indx] == -np.inf:
                continue
            else:
                combined.append([videoid, frameid, df["bb_left"][indx], df["bb_top"][indx],  df["bb_width"][indx],  df["bb_height"][indx],  df["class"][indx],  df["confidence"][indx]])
        

    fdf = pd.DataFrame(combined, columns=["video_id", "frame", "bb_left", "bb_top","bb_width", "bb_height", "class", "confidence"])



    print(fdf)
    fdf = fdf.dropna()
    fdf["class"] = fdf["class"] + 1
    fdf["bb_left"] = fdf["bb_left"]*1920 
    fdf["bb_top"] = fdf["bb_top"]*1080 
    fdf["bb_width"] = fdf["bb_width"]*1920 
    fdf["bb_height"] = fdf["bb_height"]*1080 
    fdf["bb_left"] = fdf["bb_left"] - fdf["bb_width"]/2
    fdf["bb_top"] = fdf["bb_top"] - fdf["bb_height"]/2

    fdf["bb_left"] = fdf["bb_left"].clip(lower=0, upper=1920).astype(int)
    fdf["bb_top"] = fdf["bb_top"].clip(lower=0, upper=1080).astype(int)
    fdf["bb_width"] = fdf["bb_width"].clip(lower=1, upper=1920).astype(int)
    fdf["bb_height"] = fdf["bb_height"].clip(lower=1, upper=1080).astype(int)

    fdf = fdf.sort_values(by=["video_id","frame"])

    
    fdf.to_csv(path+"vigir_submission_9.txt", 
                    index=False, 
                    header=None)
    


