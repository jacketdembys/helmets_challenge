import numpy as np
import pandas as pd
import os
import sys

if __name__ == '__main__':

    path = "results/"

    combined = []
    for i in range(1,101):
        df = pd.read_csv(path+"submission_{}.txt".format(i))
        combined.append(df)
    
    combined = pd.concat(combined)
    combined["class"] = combined["class"] + 1
    combined.to_csv(path+"vigir_submission_1.txt", 
                    index=False, 
                    header=None)

