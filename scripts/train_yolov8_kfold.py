# Import the required libraries
import datetime
import shutil
from pathlib import Path
from collections import Counter
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
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
  kf = KFold(n_splits=ksplit, shuffle=True, random_state=352023)
  kfolds = list(kf.split(labels_df))
  
  # Display the splits more clearly 
  folds = [f'split_{n}' for n in range(1, ksplit + 1)]
  folds_df = pd.DataFrame(index=indx, columns=folds)

  for idx, (train, val) in enumerate(kfolds, start=1):
    folds_df[f'split_{idx}'].loc[labels_df.iloc[train].index] = 'train'
    folds_df[f'split_{idx}'].loc[labels_df.iloc[val].index] = 'val'

  # Calculate the distribution of class labels for each fold
  fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)

  for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
    train_totals = labels_df.iloc[train_indices].sum()
    val_totals = labels_df.iloc[val_indices].sum()

    #   To avoid a division by zero, we add a small value (1E-7) to the denominator
    ratio = val_totals/(train_totals+1E-7)
    fold_lbl_distrb.loc[f'split_{n}'] = ratio

  # Create directories and dataset YAML files for each split 
  supported_extensions = ['.jpg'] # '.jpeg', '.png']

  # Initialize an empty list to store image file paths
  images  = []
  
  # Loop through supported extensions and gather image files 
  for ext in supported_extensions:
    images.extend(sorted((dataset_path / 'images').rglob(f"*{ext}")))
    print(ext)
    print(images)

  # Create the necessary directories and dataset YAML files (unchanged)
  save_path = Path(dataset_path / f'{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val')
  save_path.mkdir(parents=True, exist_ok=True)
  ds_yamls = []  
 
  for split in folds_df.columns:
    # Create directories
    split_dir = save_path / split
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / 'train' / 'images').mkdir(parents=True, exist_ok=True)
    (split_dir / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
    (split_dir / 'val' / 'images').mkdir(parents=True, exist_ok=True)
    (split_dir / 'val' / 'labels').mkdir(parents=True, exist_ok=True)

    # Create dataset YAML files
    dataset_yaml = split_dir / f'{split}_dataset.yaml'
    ds_yamls.append(dataset_yaml)

    with open(dataset_yaml, 'w') as ds_y:
      yaml.safe_dump({
        'path': split_dir.as_posix(),
        'train': 'train',
        'val':'val',
        'names': classes
      }, ds_y)


  #print(images)
  #print(labels)

  # Copy images and labels into respective directories (train, val) for each split
  for image, label in zip(images, labels):
    for split, k_split in tqdm(folds_df.loc[image.stem].items()):
        # Destination directory
        img_to_path = save_path / split / k_split / 'images'
        lbl_to_path = save_path / split / k_split / 'labels'

        # Copy image and label files to new directory (SamefileError if file already exists)
        shutil.copy(image, img_to_path / image.name)
        shutil.copy(label, lbl_to_path / label.name)

  # Save records of the K-folds split and label distribution
  folds_df.to_csv(save_path / "kfold_datasplit.csv")
  fold_lbl_distrb.to_csv(save_path / "kfold_label_distribution.csv")
  



    


