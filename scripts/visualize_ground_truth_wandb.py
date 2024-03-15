import os, cv2, torch, wandb, argparse
import pandas as pd
import numpy as np
import tqdm
import sys
from pathlib import Path
from ultralytics.utils import ops
from ultralytics.utils.plotting import plot_images


def bounding_boxes_gt(img, boxes_gt, class_name):
  
  return wandb.Image(
           img,
           #mode="RGB",
           #file_type="jpg",
           boxes={
             "groundtruth": {
               "box_data": [
                 {
                   "position": {
                     "minX": float(box[0]),
                     "minY": float(box[1]),
                     "maxX": float(box[2]),
                     "maxY": float(box[3]),
                   },
                   "class_id": int(box[4]),
                   "domain": "pixel",
                   "box_caption": class_name[int(box[4])],
                 }
                 for box in boxes_gt
               ],
               "class_labels": class_name,
             }
           },
        ) 

if __name__ == "__main__":
    
    # initialize a wandb run and table
    #run = wandb.init(project="helmets-challenge", name="ground-truth")
    parser = argparse.ArgumentParser()
    parser.add_argument("-videoid", type=str, help="video-id")
    args = parser.parse_args()
    video_id = args.videoid
    #run = wandb.init(project="helmets-challenge", 
    #                job_type="ground-truth",
    #                name=video_id)
    
    run = wandb.init(
        entity="jacketdembys",
        project = "helmets-challenge-gt",                
        group = "ground-truth",
        name = video_id,
        #job_type = "baseline"
    )
    
    table = wandb.Table(columns=["ID", "Image"])
    
    # load image and label paths
    image_paths = Path("/home/retina/dembysj/Dropbox/WCCI2024/challenges/aicity2024_track5/aicity2024_track5_train/train/images/")
    label_paths = Path("/home/retina/dembysj/Dropbox/WCCI2024/challenges/aicity2024_track5/aicity2024_track5_train/train/labels/")

    image_paths = sorted(image_paths.rglob("*.jpg"))
    label_paths = sorted(label_paths.rglob("*.txt"))

    image_paths = [i for i in image_paths if i.stem.startswith(video_id)]
    label_paths = [l for l in label_paths if l.stem.startswith(video_id)]

    #print(image_paths)
    #print(label_paths)

    # set the class names for visuaization
    class_name = {0: 'motorbike', 1: 'DHelmet', 2: 'DNoHelmet', 3: 'P1Helmet', 4: 'P1NoHelmet', 5: 'P2Helmet', 6: 'P2NoHelmet', 7: 'P0Helmet', 8: 'P0NoHelmet'}
    
    # get image count (both image and label counts are the same)
    image_count = len(image_paths)
    #print(image_count)

    #sys.exit()

    
    # process the images
    for i in range(image_count):
        print(image_paths[i])
        img = cv2.imread(str(image_paths[i])) # , cv2.IMREAD_UNCHANGED
        label = pd.read_csv(str(label_paths[i]), header=None, sep=" ")
        label = label.values.tolist()   
        bboxes = [] 
        for b in range(len(label)):
            bbox = ops.xywhn2xyxy(np.array(label[b][1:]), w=1920, h=1080).round().tolist()
            bbox.append(label[b][0])
            bboxes.append(bbox)
            #print(bboxes)
        box_img = bounding_boxes_gt(img, bboxes, class_name)      
        table.add_data(image_paths[i].stem, box_img)

        #annotate = 
        #print(label)
        #cv2.imshow(str(image_paths[0].stem), img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    #print(img)
    
    run.log({"Table": table})
    run.finish()
    