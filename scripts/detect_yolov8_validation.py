import torch
import pandas as pd
import numpy as np
import time
import sys
import wandb
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback


# compute elapsed time in mins and secs
def elapsed_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == "__main__":

	start_time = time.monotonic()

	# params to choose from
	ksplit = 1
	devices = 1
	inference_on = "images" # images / videos

	if inference_on == "videos":
		runs = [
			
		]
	

	run = wandb.init()
	artifact = run.use_artifact('jacketdembys/helmets-challenge/{}:best'.format(runs[ksplit-1]), type='results')
	artifact_dir = artifact.download()

	# Load a model
	#yolo_v8 = "n"   # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x 
	#model = YOLO("yolov8"+yolo_v8+".pt")  # load a pretrained model
	model = YOLO(artifact_dir+"/best.pt")
	device = 0 if devices == 1 else [i for i in range(devices)]
	add_wandb_callback(model)

	# Predict with loaded model
	path_data = "/home/retina/dembysj/dataset/val/"
	path_results="results/"	

	results = model.predict(
		source=path_data, #+image_folder, #+image_name,					# you can specify a video folder name containing all the extracted frames or a specific frame
		conf=0.1,
		iou=0.1,
		project=path_results,
		name= "ksplit"+str(ksplit),  #image_folder,								 
		save=True,  									# save plot result
        save_frames=True,
		save_crop=True,
		show=False,  									# show result on the screen
		save_txt=True,  								# save result in txt
		save_conf=True,  								# in result has conf score
		save_json=True,  								# save json file result
		device=device,
		#stream=True
	)  # predict on an image
      
	end_time = time.monotonic()

	mins, secs = elapsed_time(start_time, end_time)
	print('\nElapse Time: {}m {}s'.format(mins, secs))

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

	#print(results)
	#sys.exit()

	"""
	# The result is a tensor from PyTorch
	# Refactor the results in the following submission format of the AI City Challenge - Track 5
	# 〈video_id〉, 〈frame〉, 〈bb_left〉, 〈bb_top〉, 〈bb_width〉, 〈bb_height〉, 〈class〉, 〈confidence〉

	arranged_results = []
	frames = [x for x in range(0,len(results))]
	for f in frames:									# loop through the frames of the current video
		boxes = results[f].boxes						# extract the bounding boxes in each frames
		for b in range(len(boxes)):						# loop through the bounding boxes
			box = boxes[b].cpu().numpy()
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
	df.to_csv("results/submission_"+str(video_id)+".txt",
		index=False,
		header=header)


	end_time = time.monotonic()

	mins, secs = elapsed_time(start_time, end_time)
	print('\nElapse Time: {}m {}s'.format(mins, secs))
				
	#print(arranged_results)
	#print(len(results))
	#print(type(results))
	#print(results)
    """
