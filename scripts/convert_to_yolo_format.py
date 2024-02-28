import os
import numpy as np
#import cv2
import sys
import argparse
import shutil
import pandas as pd
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--gt_txt", 
                help="path to gt txt", 
                default="/home/retina/dembysj/Dropbox/WCCI2024/challenges/aicity2024_track5/aicity2024_track5_train/gt.txt")

ap.add_argument("-v", "--input_imgs_dir", 
                help="path to input video dir", 
                default="/home/retina/dembysj/Dropbox/WCCI2024/challenges/aicity2024_track5/aicity2024_track5_train/images")

ap.add_argument("-o", "--output_imgs_dir", 
                help="path to output video dir", 
                default="/home/retina/dembysj/Dropbox/WCCI2024/challenges/aicity2024_track5/aicity2024_track5_train/train/images")

ap.add_argument("-p", "--output_gt_dir", 
                help="path to part gt dir", 
                default="/home/retina/dembysj/Dropbox/WCCI2024/challenges/aicity2024_track5/aicity2024_track5_train/train/labels")

args = vars(ap.parse_args())


def convert_to_yolo_format(input_file_path, output_file_path):
    print("Ground truth conversion to YOLO format started ...")
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        for line in input_file:
            # Split the line into individual values
            video_id, frame, bb_left, bb_top, bb_width, bb_height, class_label = map(lambda x: int(x), line.strip().split(','))
            
            print(video_id, frame, bb_left, bb_top, bb_width, bb_height, class_label)

            # Convert bounding box coordinates to YOLO format
            x_center = bb_left + bb_width / 2
            y_center = bb_top + bb_height / 2
            normalized_x_center = x_center / 1920  # Assuming video width is 1920 (adjust if needed)
            normalized_y_center = y_center / 1080  # Assuming video height is 1080 (adjust if needed)
            normalized_width = bb_width / 1920
            normalized_height = bb_height / 1080
            
            # Write the YOLO-formatted line to the output file
            #output_line = f"{class_label} {normalized_x_center:.6f} {normalized_y_center:.6f} {normalized_width:.6f} {normalized_height:.6f} {class_label}\n"
            output_line = f"{class_label} {normalized_x_center:.6f} {normalized_y_center:.6f} {normalized_width:.6f} {normalized_height:.6f}\n"
            
            output_file.write(output_line)

    print("Ground truth conversion to YOLO format completed!")






def gather_images(source_folder, destination_folder):
    print("Image gathering started ...")
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    dirname_list = []

    for root, dirs, files in os.walk(source_folder):
        for dirname in dirs:
            #print(dirname)
            dirname_list.append(dirname)
    
    dirname_list.sort()

    for dirname in dirname_list:
        for root, dirs, files in os.walk(os.path.join(source_folder, dirname)):
            for filename in files:
                if filename.lower().endswith(('.jpg')):
                    source_path = os.path.join(root, filename)
                    destination_path = os.path.join(destination_folder, dirname+'_'+filename)

                    shutil.copy(source_path, destination_path)
                    #print(f"Moved {filename} from {source_path} to {destination_path}")
                    


    
        #for root, dirs, files in os.walk(os.path.join(source_folder, d))
        #print(root, dirs, files)
        #for name in files:
        #    print("here")
        #    print(name)
        #sys.exit()
    print("Image gathering completed!")
  


def check_img_label_pairs(output_img_dir, output_gt_dir):

    print("Image/label pair checking started ...")
    
    if not os.path.exists(output_img_dir+"_uncheck"):
        shutil.move(output_img_dir, output_img_dir+"_uncheck")

    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)

    for txt_file in tqdm(os.listdir(output_gt_dir)):
        if txt_file.endswith('.txt'):
            img_file = os.path.join(output_img_dir+"_uncheck", f"{os.path.splitext(txt_file)[0]}.jpg")

            #print(img_file)
            #print(output_img_dir)

            if os.path.exists(img_file):
                shutil.copy(img_file, output_img_dir)
            else:
                print(img_file)

    
    print("Image/label pair checking completed!")



def check_label_img_pairs(output_img_dir, output_gt_dir):

    print("Image/label pair checking started ...")
    
    if not os.path.exists(output_img_dir+"_uncheck"):
        shutil.move(output_img_dir, output_img_dir+"_uncheck")

    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)

    for txt_file in tqdm(os.listdir(output_gt_dir)):
        if txt_file.endswith('.jpg'):
            img_file = os.path.join(output_img_dir+"_uncheck", f"{os.path.splitext(txt_file)[0]}.txt")

            #print(img_file)
            #print(output_img_dir)

            if os.path.exists(img_file):
                shutil.copy(img_file, output_img_dir)
            else:
                print(img_file)

    
    print("Image/label pair checking completed!")




def convert_to_YOLO_format(input_file_path, output_file_path, output_img_dir):
    print("Ground truth conversion to YOLO format started ...")
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
  
    
    previous_frame = 0

    with open(input_file_path, 'r') as input_file:
        for line in input_file:
            # Split the line into individual values
            video_id, frame, bb_left, bb_top, bb_width, bb_height, class_label = map(lambda x: int(x), line.strip().split(','))
            

            print(video_id, frame, bb_left, bb_top, bb_width, bb_height, class_label)

            # Convert bounding box coordinates to YOLO format
            x_center = bb_left + bb_width / 2
            y_center = bb_top + bb_height / 2
            normalized_x_center = x_center / 1920  # Assuming video width is 1920 (adjust if needed)
            normalized_y_center = y_center / 1080  # Assuming video height is 1080 (adjust if needed)
            normalized_width = bb_width / 1920
            normalized_height = bb_height / 1080

            if normalized_x_center > 1.0 or normalized_y_center > 1.0 or normalized_width > 1.0 or normalized_height > 1.0:
                continue

            if frame > 200: # 200 is the maximum number of frames in a video
                continue

            print(x_center, y_center)

            if len(str(video_id)) == 1:
                video_id_name = '00'+str(video_id)
            elif len(str(video_id)) == 2:
                video_id_name = '0'+str(video_id)
            elif len(str(video_id)) == 3:
                video_id_name = str(video_id)
                        
            if len(str(frame)) == 1:
                frame_name = '0000000'+str(frame)
            elif len(str(frame)) == 2:
                frame_name = '000000'+str(frame)
            elif len(str(frame)) == 3:
                frame_name = '00000'+str(frame)
            
            # Write the YOLO-formatted line to the output file
            #output_line = f"{class_label} {normalized_x_center:.6f} {normalized_y_center:.6f} {normalized_width:.6f} {normalized_height:.6f} {class_label}\n"
            #output_line = f"{class_label} {normalized_x_center:.6f} {normalized_y_center:.6f} {normalized_width:.6f} {normalized_height:.6f}\n"
            
            if frame == previous_frame:
                output_line = f"{class_label-1} {normalized_x_center:.6f} {normalized_y_center:.6f} {normalized_width:.6f} {normalized_height:.6f}\n"
            
                new_output_file_path = os.path.join(output_file_path, video_id_name+'_'+frame_name+'.txt')
                print(new_output_file_path)
                print()

                with open(new_output_file_path, 'a') as output_file:
                    output_file.write(output_line)
            else:
                output_line = f"{class_label-1} {normalized_x_center:.6f} {normalized_y_center:.6f} {normalized_width:.6f} {normalized_height:.6f}\n"
                

                new_output_file_path = os.path.join(output_file_path, video_id_name+'_'+frame_name+'.txt')
                print(new_output_file_path)
                print()

                with open(new_output_file_path, 'w') as output_file:
                    output_file.write(output_line)

                previous_frame = frame


    print("Ground truth conversion to YOLO format completed!")



def gather_YOLO_images(source_folder, gt_folder, destination_folder):
    print("Image gathering started ...")
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    dirname_list = []

    # Get the list of directory in image folder
    for root, dirs, files in os.walk(source_folder):
        for dirname in dirs:
            #print(dirname)
            dirname_list.append(dirname)
    
    # sort the directories by ascending oder
    dirname_list.sort()

    # Get the list of ground truth files in ground truth folder
    gt_file_list = []
    for root, dirs, files in os.walk(gt_folder):
        for file in files:
            gt_file_list.append(file)

    # sort the files by ascending oder
    gt_file_list.sort()
    

    # gather the images from image directory by checking if the corresponding ground truth file exists
    for dirname in dirname_list:
        for root, dirs, files in os.walk(os.path.join(source_folder, dirname)):
            for filename in files:
                if filename.lower().endswith(('.jpg')):
                    source_path = os.path.join(root, filename)
                    destination_path = os.path.join(destination_folder, dirname+'_'+filename)
                    img_file = f"{dirname}_{os.path.splitext(filename)[0]}.txt"

                    if img_file in gt_file_list:
                        print(f"Moved {filename} from {source_path} to {destination_path}")
                        shutil.copy(source_path, destination_path)
                        
                    
    print("Image gathering completed!")




def main():
    
    gt_txt = args['gt_txt']
    path_img_dir = args["input_imgs_dir"]
    paths_img = os.listdir(path_img_dir)
    output_img_dir = args['output_imgs_dir']    
    output_gt_dir = args['output_gt_dir']
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    

    #print(gt_txt)
    #print(output_img_dir)

    # convert current gt format to YOLO format
    #convert_to_yolo_format(gt_txt, os.path.join(output_img_dir, 'converted_gt_txt.txt'))

    #convert_to_yolo_format_2(gt_txt, output_gt_dir)

    # gather images in one folder
    #gather_images(path_img_dir, output_img_dir)

    # check the img and label pairs
    #check_img_label_pairs(output_img_dir, output_gt_dir)


    #check_label_img_pairs(output_gt_dir, output_img_dir)









    ## Load from the groundtruth file 
    # 〈video_id〉, 〈frame〉, 〈bb_left〉, 〈bb_top〉, 〈bb_width〉, 〈bb_height〉, 〈class〉
    gt = pd.read_csv(gt_txt, header=None)
    print(len(gt))

    ## Convert to correct YOLO format
    #convert_to_YOLO_format(gt_txt, output_gt_dir, output_img_dir)

    ## Check that the images
    gather_YOLO_images(path_img_dir, output_gt_dir, output_img_dir)
    

if __name__ == '__main__':
    main()