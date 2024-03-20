import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from cf_matrix import make_confusion_matrix
from sklearn import metrics


def plot_confusion_matrix(run_ids, categories, figsize=(12,8), block_fig=True):

    # download confusion matrix from split 1
    nclass = len(categories)
    df = pd.DataFrame(0, index=range(nclass), columns=range(nclass))
    for i in run_ids:
        artifact = run.use_artifact('jacketdembys/helmets-challenge/run_{}_results:v1'.format(i), type='results')
        artifact_dir = artifact.download()
        cm = pd.read_csv(os.path.join(artifact_dir, "confusion_matrix.csv"), header=None)
        df = (df + cm)

    df = df / 10
    dfa = np.array(df) 
    #df.columns = ["motorbike (T)", "person (T)", "background (T)"]
    #df.index = ["motorbike (p)", "person", "background"]
    print(df)
    print(dfa)

    #labels = ['True Neg','False Pos','False Neg','True Pos']
    #categories = ['motorbike', 'person', 'background']
    make_confusion_matrix(dfa, 
                        #group_names=labels,
                        categories=categories,
                        figsize=figsize 
                        #cmap=’binary’
                        )
    plt.savefig("confusion_matrix_{}class_cross_validation.pdf".format(nclass-1), format="pdf", bbox_inches="tight")
    plt.show(block=block_fig)


if __name__ == "__main__":

    # log in to wandb
    run = wandb.init()

    # list of run_id for different splits for the 2 class cross-validation
    run_id_2class = ["ksis0pj5", # split 1
                    "xiqyx1av", # split 2
                    "f7c24dkl", # split 3
                    "qr8p4ltx", # split 4
                    "08683tur", # split 5
                    "4je3rh8a", # split 6
                    "1o696x65", # split 7
                    "6hmm1u6r", # split 8
                    "56cch60b", # split 9
                    "41loqzhl", # split 10
                    ]

    categories = ['motorbike', 'person', 'background']
    plot_confusion_matrix(run_id_2class, categories, figsize=(12,8))



    # list of run_id for different splits for the 2 class cross-validation
    run_id_3class = ["3rr822cf", # split 1
                    "fg245qnf", # split 2
                    "26nz6ou8", # split 3
                    "p82dqk8f", # split 4
                    "ftrfk9vv", # split 5
                    "xl13sscc", # split 6
                    "kkxb4mgf", # split 7
                    "qam05520", # split 8
                    "ygx9rk5y", # split 9
                    "naq7pgfp", # split 10
                    ]

    categories = ['motorbike', 'personHelmet','personNoHelmet', 'background']
    plot_confusion_matrix(run_id_3class, categories, figsize=(12,8))

