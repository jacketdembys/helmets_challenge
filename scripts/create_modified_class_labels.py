import numpy as np
import pandas as pd
import os, sys, argparse
import shutil
from pathlib import Path
from tqdm import tqdm

"""
# 2-class model - Keep only class motorbike and persons
0: motorbike
1: person (DHelmet, DNoHelmet, P1Helmet, P1NoHelmet, P2Helmet, P2NoHelmet, P0Helmet, P0NoHelmet)

# 3-class model
0 : motorbike
1 : PersonHelmet
2 : PersonNoHelmet

# 5-class model
0: motorbike
1: DriverHelmet
2: DriverNoHelmet
3: PassengerHelmet
4: PassengerNoHelmet
"""

if __name__ == "__main__":
    print("Modify the class labels ...")

    parser = argparse.ArgumentParser(description="yolov8x helmet experiment")
    parser.add_argument('-nc', type=int, default=2,  help="number of classes in the corrected labels")
    args = parser.parse_args()

    num_class = args.nc

    #path = "../../aicity2024_track5/dataset_"+str(num_class)+"class/"
    path = "/home/dataset/"
    dataset_path = Path(path)
    labels = sorted(dataset_path.rglob("*labels/*.txt"))

    #print(len(labels))

    for i in tqdm(labels):
        #print(i)
        #print(i.stem)
        #print(i.name)

        df = pd.read_csv(i, header=None, sep=" ")

        if num_class == 2:
            df.loc[df.iloc[:, 0] > 1, 0] = 1

        if num_class == 3:
            inds = [3, 5, 7]
            for idx in inds:
                df.loc[df.iloc[:, 0] == idx, 0] = 1
                df.loc[df.iloc[:, 0] == idx+1, 0] = 2

        if num_class == 5:
            inds = [3, 5, 7]
            for idx in inds:
                df.loc[df.iloc[:, 0] == idx, 0] = 3
                df.loc[df.iloc[:, 0] == idx+1, 0] = 4

        df.to_csv(i, 
                index=False, 
                header=None,
                sep=" ")
        

    print("Modify the class labels done!")