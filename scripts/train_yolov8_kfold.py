# Import the required libraries
import datetime
import shutil
import yaml
import numpy as np
import pandas as pd
import time
import torch, json, wandb, contextlib, argparse
import torch.nn as nn
import ultralytics.nn.tasks as tasks
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import KFold
from ultralytics.utils import LOGGER, RANK, colorstr
from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight
from ultralytics.data.augment import Albumentations
from ultralytics.utils.torch_utils import make_divisible
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C3,
    C3TR,
    OBB,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    ImagePoolingAttn,
    C3Ghost,
    C3x,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    Pose,
    RepC3,
    RepConv,
    ResNetLayer,
    RTDETRDecoder,
    Segment,
    WorldDetect,
)


# Build a custom YOLO model
def load_model_custom(self, cfg=None, weights=None, verbose=True):
  """Return a YOLO detection model."""
  weights, _ = attempt_load_one_weight("yolov8l.pt") 
  model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
  if weights:
    model.load(weights)
  return model


# Callback function to log out the confusion matrix
def save_confusion_matrix(validator):
  confusion_matrix = validator.confusion_matrix.matrix
  confusion_matrix = pd.DataFrame(confusion_matrix)
  save_path = "confusion_matrix.csv"
  confusion_matrix.to_csv(
    save_path, 
    index=False, 
    header=None
  )
  artifact = wandb.Artifact(type="results", name=f"run_{wandb.run.id}_results")
  artifact.add_file(local_path=save_path)
  wandb.run.log_artifact(artifact)  
  #print("\n#### debug: {}\n".format(confusion_matrix))




if __name__ == '__main__':

  parser = argparse.ArgumentParser(description="yolov8 helmet experiment")
  parser.add_argument('-config', type=str, default="helmet_data.yaml",  help="config file for model training")
  parser.add_argument('-devices', type=int, default=1,  help="number of gpus")
  parser.add_argument('-epochs',   type=int, default=10,  help="number of epoch")
  parser.add_argument('-bs',      type=int, default=16, help="number of batches")
  parser.add_argument('-imgsz',   type=int, default=640, help="resize image before feeding to model")
  parser.add_argument('-rpath',   type=str, default="/home/results/", help="path to results")
  parser.add_argument('-name',    type=str,   default="yolov8l-xval", help="run name")
  parser.add_argument('-project', type=str,   default="helmets-challenge", help="project name")
  parser.add_argument('-frac',    type=float, default=1.0, help="fraction of the data being used")
  parser.add_argument('-csplit',    type=int, default=1, help="chosen k split to train on for multi-resource cross validation")
  args = parser.parse_args()

  print("Prepare dataset for cross validation ...")

  st = time.time()
    
  # Retrieve all labels for the dataset
  dataset_path = Path('/home/dataset')
  labels = sorted(dataset_path.rglob("*labels/*.txt"))
  #print(labels)
  #print(len(labels))

  # Read the content of the YAML file
  yaml_file = args.config #'helmet_data.yaml'
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
  #labels = sorted(dataset_path.rglob("*labels/*.txt"))
  images = sorted(dataset_path.rglob("*images/*.jpg"))
  """
  for ext in supported_extensions:
    images.extend(sorted((dataset_path / '*images/').rglob(f"*{ext}")))
    print(dataset_path / 'images')
    print((dataset_path / 'images').rglob(f"*{ext}"))
    print(ext)
    print(images)
  """

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


  #print(images[:5])
  #print(labels[:5])

  
  # Copy images and labels into respective directories (train, val) for each split
  for image, label in zip(images, labels):
    for split, k_split in folds_df.loc[image.stem].items():
        # Destination directory
        img_to_path = save_path / split / k_split / 'images'
        lbl_to_path = save_path / split / k_split / 'labels'

        # Copy image and label files to new directory (SamefileError if file already exists)
        shutil.copy(image, img_to_path / image.name)
        shutil.copy(label, lbl_to_path / label.name)

  # Save records of the K-folds split and label distribution
  folds_df.to_csv(save_path / "kfold_datasplit.csv")
  fold_lbl_distrb.to_csv(save_path / "kfold_label_distribution.csv")


  et = time.time()

  print("Done preparing the dataset!")

  # Train YOLOv8 using K-Fold Data Split
  DetectionTrainer.get_model = load_model_custom

  device = 0 if args.devices == 1 else [i for i in range(args.devices)]

  print(type(shutil.which(ds_yamls[args.csplit-1])))
  print(type(ds_yamls[args.csplit-1]))

  train_args = dict(project=args.project, 
                    name=args.name,
                    model="yolov8l.yaml", 
                    data= ds_yamls[args.csplit-1].path, #args.config,
                    device=device, 
                    epochs=args.epochs, 
                    batch=args.bs, 
                    fraction=args.frac, 
                    imgsz=args.imgsz,
                    exist_ok=True,
                    val=True, 
                    #freeze=10,
                    #save_json=True, 
                    conf=0.001, 
                    #iou=0.5,
                    #lr0=0.001,
                    #optimizer="AdamW", 
                    #seed=0,
                    #box=7.5, 
                    #cls=0.125, 
                    #dfl=3.0,
                    #close_mosaic=0,
                    hsv_h=0.5, # (float) image HSV-Hue augmentation (fraction)
                    hsv_s=0.8, # (float) image HSV-Saturation augmentation (fraction)
                    hsv_v=0.8, # (float) image HSV-Value augmentation (fraction)
                    degrees=10.0, # (float) image rotation (+/- deg)
                    translate=0.1, # (float) image translation (+/- fraction)
                    scale=0.8, # (float) image scale (+/- gain)
                    shear=0.0, # (float) image shear (+/- deg)
                    perspective=0.0001, # (float) image perspective (+/- fraction), range 0-0.001
                    flipud=0.8, # (float) image flip up-down (probability)
                    fliplr=0.5, # (float) image flip left-right (probability)
                    mosaic=1, # (float) image mosaic (probability)
                    mixup=0.0, # (float) image mixup (probability)
                    copy_paste=0.0, # (float) segment copy-paste (probability)
                    auto_augment="randaugment", # (str) auto augmentation policy for classification (randaugment, autoaugment, augmix)
                    erasing=0.6, # (float) probability of random erasing during classification training (0-1)
                    )
  
  trainer = DetectionTrainer(overrides=train_args)
  trainer.add_callback("on_val_end", save_confusion_matrix)
  trainer.train()
    


