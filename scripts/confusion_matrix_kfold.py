import wandb
import numpy as np
import pandas as pd
import os

# log in to wandb
run = wandb.init()

# list of run_id for different splits
run_id = ["ksis0pj5", # split 1
          "xiqyx1av", # split 2
          "f7c24dkl", # split 3
          "qr8p4ltx", # split 4
          "qr8p4ltx", # split 5
          ]
# download confusion matrix from split 1
for i in run_id:
    artifact = run.use_artifact('jacketdembys/helmets-challenge/run_{}_results:v1'.format(i), type='results')
    artifact_dir = artifact.download()
    cm = pd.read_csv(os.path.join(artifact_dir, "confusion_matrix.csv"), header=None)
    print(cm)




