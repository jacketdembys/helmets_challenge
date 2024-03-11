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
  print(labels_df)