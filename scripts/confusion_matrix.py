import numpy as np 
import pandas as pd
import torch
import math
import warnings
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics.utils.metrics import ConfusionMatrix, batch_probiou, box_iou
from ultralytics.utils import ops
from ultralytics.utils import LOGGER, SimpleClass, TryExcept, plt_settings



if __name__ == "__main__":
    ## Load the ground truth
    video_id = "009"
    labels_path = Path("../../aicity2024_track5/dataset_all/train/labels/"+video_id+"/")
    labels_path = sorted(labels_path.rglob("*.txt"))

    gt_labels = []
    counts = {i: 0 for i in range(10)}
    for i in labels_path:
        df = pd.read_csv(i, header=None, sep=" ")
        #print()
        for j in range(10):
            ss = df[df[0] == j].shape[0]
            counts[j] += ss    
            #print(df[df[0] == j].shape[0])
        #print("frameid: ", i)
        #print("df:\n", df)

        gt_labels.append(df)

    labels = pd.concat(gt_labels)
    tcids = np.array(labels[0])


    print(tcids.shape)
    print(counts)
    print(sum(counts.values()))
    #sys.exit()


    ## Load the predictions without postprocessing
    #path = "results_cm_3/"+video_id+"/"
    #path = "results_cm_3/"+video_id+"_pp1/"
    #path = "results_cm_3/"+video_id+"_pp2/"
    #path = "results_cm_027/"+video_id+"/"
    path = "results_rs/"+video_id+"_pp2/"
    #path = "results_pp1/"+video_id+"/"
    prwop_labels_path = Path(path)
    prwop_labels_path = sorted(prwop_labels_path.rglob("*.txt"))
    labels_path_int = [int(i.stem.split("_")[1]) for i in labels_path]

    prwop_labels = []
    #count = 1
    count = {}
    for i in prwop_labels_path:
        cur =  int(i.stem.split("_")[1])
        if cur in labels_path_int:  
            df = pd.read_csv(i, header=None, sep=" ")
            prwop_labels.append(df)

            for j in range(10):
                ss = df[df[0] == j].shape[0]
                counts[j] += ss    
            
            #count += 1
    print(counts)
    print(sum(counts.values()))

    prwop_labels = pd.concat(prwop_labels)
    prwop_cids = np.array(prwop_labels[0])    

      

    ## Compute the confusion matrix between the gt and pr without postprocessing
    sw_labels = np.array(labels)
    gt_labels = sw_labels[:,0]
    gt_bboxes = sw_labels[:,1:]

    dt_labets = np.array(prwop_labels)
    dt_labets = np.concatenate((dt_labets[:, 1:6], dt_labets[:, 0].reshape(-1, 1)), axis=1)

    gt_bboxes = torch.from_numpy(gt_bboxes)
    gt_labels = torch.from_numpy(gt_labels)
    dt_labets = torch.from_numpy(dt_labets)

    print(gt_bboxes.shape)
    print(dt_labets.shape)
    print(gt_labels.shape)

    for i in range(gt_bboxes.shape[0]):
        gt_bboxes[i,:] = ops.xywh2xyxy(gt_bboxes[i,:])

    for j in range(dt_labets.shape[0]):
        dt_labets[j,:4] = ops.xywh2xyxy(dt_labets[j,:4])

    #cm = ConfusionMatrix(nc=9, conf=0.1, iou_thres=0.1, task="detect")
    cm = ConfusionMatrix(nc=9, task="detect")
    cm.process_batch(dt_labets, gt_bboxes=gt_bboxes, gt_cls=gt_labels)
    #print(cm.matrix)

    categories = ('motorbike', 'DHelmet','DNoHelmet','P1Helmet','P1NoHelmet','P2Helmet','P2NoHelmet','P0Helmet','P0NoHelmet')
    cm.plot(normalize=False, names=categories, save_dir=path)
    
    """
    categories = ['motorbike', 'DHelmet','DNoHelmet','P1Helmet','P1NoHelmet','P2Helmet','P2NoHelmet','P0Helmet','P0NoHelmet', 'background']
    cm = pd.DataFrame(cm.matrix)
    cm.columns = categories
    cm.index = categories
    print(cm)
    """
    









