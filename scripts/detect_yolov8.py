import torch
import pandas as pd
import numpy as np
from ultralytics import YOLO

# params to choose from
inference_on = "images" # images / videos

# Load a model
yolo_v8 = "n"   # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x 
model = YOLO("yolov8"+yolo_v8+".pt")  # load a nano model

# Predict with loaded model
path_data = "/home/retina/dembysj/Dropbox/WCCI2024/challenges/aicity2024_track5_train/aicity2024_track5_train/images/"
path_results="/home/retina/dembysj/Dropbox/WCCI2024/challenges/ETSS-01-Edge-TSS/src/aic23/track_5/"
video_id = 2
image_folder = "00"+str(video_id)+"/"
image_name = "00000100.jpg" 
results = model.predict(
	source=path_data+image_folder+image_name,					# you can specify a video folder name containing all the extracted frames or a specific frame
	conf=0.25,
	project=path_results+"results/",
	name=image_folder,								 
	save=True,  									# save plot result
	save_crop=True,
	show=True,  									# show result on the screen
	save_txt=True,  								# save result in txt
	save_conf=True,  								# in result has conf score
	save_json=True  								# save json file result
)  # predict on an image

"""
results = model(
	source="/home/retina/dembysj/Dropbox/WCCI2024/challenges/aicity2024_track5_train/"
		   "aicity2024_track5_train/videos/"
		   "002.mp4",
	conf=0.25,  # 0.001
	save=True,
	show=True,
)  # predict on a video
"""



# The result is a tensor from PyTorch
# Refactor the results in the following submission format of the AI City Challenge - Track 5
# 〈video_id〉, 〈frame〉, 〈bb_left〉, 〈bb_top〉, 〈bb_width〉, 〈bb_height〉, 〈class〉, 〈confidence〉

arranged_results = []
frames = [x for x in range(0,len(results))]
for f in frames:									# loop through the frames of the current video
	boxes = results[f].boxes						# extract the bounding boxes in each frames
	for b in range(len(boxes)):						# loop through the bounding boxes
		box = boxes[b].numpy()
		xywh = box.xywh
		bb_left = xywh[0,0]
		bb_top = xywh[0,1]
		bb_width = xywh[0,2]
		bb_height = xywh[0,3]
		detected_cls = box.cls.item()
		detected_conf = box.conf.item()

		arranged_results.append([video_id, f+1, bb_left, bb_top, bb_width, bb_height, int(detected_cls), detected_conf])


# Save evluation results for challenge submission
header = ["video_id", "frame", "bb_left", "bb_top","bb_width", "bb_height", "class", "confidence"]

df = pd.DataFrame(np.array(arranged_results))
df.to_csv(path_results+"results/submission.txt",
	index=False,
	header=header)


#print(arranged_results)
#print(len(results))
#print(type(results))
#print(results)
