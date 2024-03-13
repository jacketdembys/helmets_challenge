import sys
import tqdm
from aicityevalhelmet import *
from pathlib import Path



"""
# Process the predictions
path_1 = "results/test_val_images/labels/"
path_3 = "results/test_val_images/"
dataset_path = Path(path_1)
predicted_labels = sorted(dataset_path.rglob("*.txt"))

combined = []
for i in range(len(predicted_labels)):
    filename = predicted_labels[i].stem
    filename = filename.split("_")
    videoid = int(filename[0])
    frameid = int(filename[1])

    print("Prediction ... Processing video id [{}] - frame id [{}]".format(videoid, frameid))

    df = pd.read_csv(predicted_labels[i], header=None, sep=" ")
    df.columns = ["class", "bb_left", "bb_top","bb_width", "bb_height", "confidence"]
    
    for indx in df.index:
        combined.append([videoid, frameid, df["bb_left"][indx], df["bb_top"][indx],  df["bb_width"][indx],  df["bb_height"][indx],  df["class"][indx],  df["confidence"][indx]])
    

fdf = pd.DataFrame(combined, columns=["video_id", "frame", "bb_left", "bb_top","bb_width", "bb_height", "class", "confidence"])

fdf["class"] = fdf["class"] + 1
fdf["bb_left"] = fdf["bb_left"]*1920 
fdf["bb_top"] = fdf["bb_top"]*1080 
fdf["bb_width"] = fdf["bb_width"]*1920 
fdf["bb_height"] = fdf["bb_height"]*1080 
fdf["bb_left"] = fdf["bb_left"] - fdf["bb_width"]/2
fdf["bb_top"] = fdf["bb_top"] - fdf["bb_height"]/2

fdf["bb_left"] = fdf["bb_left"].clip(lower=0, upper=1920)
fdf["bb_top"] = fdf["bb_top"].clip(lower=0, upper=1080)
fdf["bb_width"] = fdf["bb_width"].clip(lower=1, upper=1920)
fdf["bb_height"] = fdf["bb_height"].clip(lower=1, upper=1080)

fdf.to_csv(path_3+"val_predictions.txt", 
                index=False, 
                header=None)


# Process the ground truth
path_2 = "/home/retina/dembysj/Dropbox/WCCI2024/challenges/aicity2024_track5/dataset/val/labels/"
dataset_path = Path(path_2)
predicted_labels = sorted(dataset_path.rglob("*.txt"))

combined = []
for i in range(len(predicted_labels)):
    filename = predicted_labels[i].stem
    filename = filename.split("_")
    videoid = int(filename[0])
    frameid = int(filename[1])

    print("Ground truth ... Processing video id [{}] - frame id [{}]".format(videoid, frameid))

    df = pd.read_csv(predicted_labels[i], header=None, sep=" ")
    df.columns = ["class", "bb_left", "bb_top","bb_width", "bb_height"]
    
    for indx in df.index:
        combined.append([videoid, frameid, df["bb_left"][indx], df["bb_top"][indx],  df["bb_width"][indx],  df["bb_height"][indx],  df["class"][indx]])
    
fdf = pd.DataFrame(combined, columns=["video_id", "frame", "bb_left", "bb_top","bb_width", "bb_height", "class"])
fdf["class"] = fdf["class"] + 1
fdf["bb_left"] = fdf["bb_left"]*1920 
fdf["bb_top"] = fdf["bb_top"]*1080 
fdf["bb_width"] = fdf["bb_width"]*1920 
fdf["bb_height"] = fdf["bb_height"]*1080 
fdf["bb_left"] = fdf["bb_left"] - fdf["bb_width"]/2
fdf["bb_top"] = fdf["bb_top"] - fdf["bb_height"]/2
fdf.to_csv(path_3+"val_ground_truth.txt", 
                index=False, 
                header=None)

"""
gt = getData('results/val_images_0.25/val_ground_truth.txt')
pr = getData('results/val_images_0.25/val_predictions.txt')
validate(pr)
m_ap = compute_map(gt, pr)
print(m_ap)

