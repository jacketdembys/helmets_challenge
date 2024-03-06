"""
We create a custom validation set by extracting few frames from each 
video to have a sample-wise validation and not a video-wise validation
"""
import numpy as np
import pandas as pd
import os
import sys
import argparse
import random
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_imgs_dir", 
                help="path to output video dir", 
                default="/home/retina/dembysj/Dropbox/WCCI2024/challenges/aicity2024_track5/dataset/train/images")

ap.add_argument("-l", "--input_gt_dir", 
                help="path to part gt dir", 
                default="/home/retina/dembysj/Dropbox/WCCI2024/challenges/aicity2024_track5/dataset/train/labels")

ap.add_argument("-vi", "--val_imgs_dir", 
                help="path to output video dir", 
                default="/home/retina/dembysj/Dropbox/WCCI2024/challenges/aicity2024_track5/dataset/val/images")

ap.add_argument("-vl", "--val_gt_dir", 
                help="path to part gt dir", 
                default="/home/retina/dembysj/Dropbox/WCCI2024/challenges/aicity2024_track5/dataset/val/labels")

args = vars(ap.parse_args())



def create_validation_set(t_imgs, t_labs, v_imgs, v_labs):
    print("Creation of validation set started ...")
    if not os.path.exists(v_imgs):
        os.makedirs(v_imgs)

    if not os.path.exists(v_labs):
        os.makedirs(v_labs)

    # Get the list of directories in image folder
    files_list = []
    for root, dirs, files in os.walk(t_imgs):
        files_list.append(files)
        #print()
        #print(root)
        #print(dirs)
    files_list = files_list[0]
    files_list.sort()
    #print(files_list)

    # Loop through the filename and randomly move some images
    # in the custom validation set
    for i in range (1,101):
        
        print()
        # Get the prefix of the video
        if len(str(i)) == 1:
            substring = "00"+str(i)
        elif len(str(i)) == 2:
            substring = "0"+str(i)
        elif len(str(i)) == 3:
            substring = str(i)

        # Count the number of files starting with the current prefix
        current_frames = [s for s in files_list if s.startswith(substring)]
        count = len(current_frames)
        #val_choice = [random.randint(1,count) for _ in range(5)]
        val_choice = random.sample(current_frames, k=5)
        print(val_choice)

        # Move the randomly chosen validation images into the validation set
        for f in val_choice:

            # Move the image
            image_name = f
            image_path = os.path.join(t_imgs, image_name)
            print(f"Moved {image_name} from {t_imgs} to {v_imgs}")
            shutil.move(image_path, v_imgs)

            # Move the label
            label_name = f.replace(".jpg", ".txt")
            label_path = os.path.join(t_labs, label_name)
            print(f"Moved {label_name} from {t_labs} to {v_labs}")
            shutil.move(label_path, v_labs)
            #sys.exit()

        #print(current_frames)
        #print(val_choice)
        #sys.exit()
   
    print("Creation of validation set done!")
        







if __name__ == "__main__":

    t_imgs = args["input_imgs_dir"]         # training images
    t_labs = args["input_gt_dir"]           # training labels
    v_imgs = args["val_imgs_dir"]           # validation images
    v_labs = args["val_gt_dir"]             # validation labels

    create_validation_set(t_imgs, t_labs, v_imgs, v_labs)
