import numpy as np
import pandas as pd
import os, sys, argparse
import shutil
from pathlib import Path
from tqdm import tqdm

"""
# Keep only class motorbike and persons
0: motorbike
1: person (DHelmet, DNoHelmet, P1Helmet, P1NoHelmet, P2Helmet, P2NoHelmet, P0Helmet, P0NoHelmet,)
"""

if __name__ == "__main__":
    print("Modify the class labels ...")


    #path = "../../aicity2024_track5/dataset_ensemble/"
    path = "/home/dataset/"
    dataset_path = Path(path)
    labels = sorted(dataset_path.rglob("*labels/*.txt"))

    #print(len(labels))

    for i in tqdm(labels):
        #print(i)
        #print(i.stem)
        #print(i.name)

        df = pd.read_csv(i, header=None, sep=" ")
        df.loc[df.iloc[:, 0] > 1, 0] = 1

        df.to_csv(i, 
                index=False, 
                header=None,
                sep=" ")
        

        #sys.exit()