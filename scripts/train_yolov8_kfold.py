# Import the required libraries
import datetime
import shutil
from pathlib import Path
from collections import Counter
import yaml
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sklearn.model_selection import KFold


if __name__ == '__main__':
    
  # Retrieve all labels for the dataset
  dataset_path = Path('/home/dataset')
  labels = sorted(dataset_path.rglob("*labels/*.txt"))
  #print(labels)

  # Read the content of the YAML file
  yaml_file = 'helmet_data.yaml'
  with open(yaml_file, 'r', encoding="utf8") as y:
    classes = yaml.safe_load(y)['names']
  cls_idx = sorted(classes.keys())
  #print(cls_idx)

  # Initialize an empty pandas dataframe
  indx = [l.stem for l in labels]           # use base filename as ID (no extension)
  labels_df = pd.DataFrame([], columns=cls_idx, index=indx)
  #print(labels_df)

  # Count the instance of each class label present in the annotation files
  for label in labels:
    lbl_counter = Counter()

    with open(label, 'r') as lf:
      lines = lf.readlines()

    for l in lines:
      # Classes for YOLO label uses integer at first position of each line
      lbl_counter[int(l.split(' ')[0])] += 1

    labels_df.loc[label.stem] = lbl_counter

  # Replace 'nan' values with '0.0'
  labels_df = labels_df.fillna(0.0)
  #print(labels_df)


  # K-Fold dataset split (setting random_state for repeatable results)
  ksplit = 10
  kf = KFold(n_splits=ksplit, 
             shuffle=True, 
             random_state=352023)

  kfolds = list(kf.split(labels_df))
  print(kfolds)