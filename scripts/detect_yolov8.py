import torch
import pandas as pd
import numpy as np
import time
import argparse
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback


# Steps for submission
"""
1. detect_yolov9.py
2. generate_final_submissions.py
"""



# compute elapsed time in mins and secs
def elapsed_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="yolov8x helmet detection")
	parser.add_argument('-sidx', type=int, default=1,  help="start index")
	parser.add_argument('-eidx', type=int, default=100,  help="end index")
	args = parser.parse_args()

	start_time = time.monotonic()

	# params to choose from
	inference_on = "images" # images / videos
	devices = 1

	# Load a model
	#yolo_v8 = "n"   # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x 
	#model = YOLO("yolov8"+yolo_v8+".pt")  # load a pretrained model
	model = YOLO("../../aicity2024_track5/weights/yolov8l-increase-augment.pt")

	device = 0 if devices == 1 else [i for i in range(devices)]
	#add_wandb_callback(model)

	# Predict with loaded model
	path_data = "/home/retina/dembysj/Dropbox/WCCI2024/challenges/aicity2024_track5/aicity2024_track5_test/test/"
	#path_data = "/home/retina/dembysj/Dropbox/WCCI2024/challenges/aicity2024_track5/dataset/val/"
	#path_results="/home/retina/dembysj/Dropbox/WCCI2024/challenges/aicity2024_track5/aicity2024_track5_test/"
	path_results="results_yolov8l_increase_augment"	
	#folder = "test_images_change_augment_05/"

	# Loop through all the video folders
	for id in range(args.sidx, args.eidx+1):

		video_id = id
		if len(str(id)) == 1:
			image_folder = "00"+str(video_id)+"/"
		elif len(str(id)) == 2:
			image_folder = "0"+str(video_id)+"/"
		elif len(str(id)) == 3:
			image_folder = str(video_id)+"/"
			
		#image_name = "00000100.jpg" 
		results = model.predict(
			source=path_data+image_folder, #+image_name,					# you can specify a video folder name containing all the extracted frames or a specific frame
			conf=0.001,
			project=path_results,
			name=id,  #image_folder,								 
			save=True,  									# save plot result
			save_crop=True,
			show=False,  									# show result on the screen
			save_txt=True,  								# save result in txt
			save_conf=True,  								# in result has conf score
			save_json=True,  								# save json file result
			device=device,
			#batch=64,
			#stream=True
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

		#print(results)


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
		df.to_csv(path_results+"/submission_"+str(video_id)+".txt",
			index=False,
			header=header)


		end_time = time.monotonic()

		mins, secs = elapsed_time(start_time, end_time)
		print('\nElapse Time: {}m {}s'.format(mins, secs))
					
		#print(arranged_results)
		#print(len(results))
		#print(type(results))
		#print(results)
