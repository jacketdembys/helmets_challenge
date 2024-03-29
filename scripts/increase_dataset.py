import sys
import shutil
import os
from pathlib import Path



dataset_uncheck_path = Path("../../aicity2024_track5/aicity2024_track5_train/train/old/train_uncheck/")
images_uncheck = sorted(dataset_uncheck_path.rglob("*.jpg"))
dataset_check_path = Path("../../aicity2024_track5/aicity2024_track5_train/train/images/")
images_check = sorted(dataset_check_path.rglob("*.jpg"))

images_uncheck_stem = [i.name for i in images_uncheck]
images_check_stem = [i.name for i in images_check]

if os.p

for i, v in enumerate(images_uncheck_stem):
    print(i, v)
    if i in images_check_stem:
        continue
    else:
        dest = 
        shutil.copy(images_uncheck_stem[i], dest)
    sys.exit()