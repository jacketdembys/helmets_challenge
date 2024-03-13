import pandas as pd

# load the submission file with x,y center coordinates of bounding boxes
fdf = pd.read_csv("results/vigir_submission_1_0.25/vigir_submission_1_old.txt", header=None)
fdf.columns = ["video_id", "frame", "bb_left", "bb_top","bb_width", "bb_height", "class", "confidence"]
print(fdf)

# convert x,y bb center to x,y left top 
#fdf["bb_left"] = fdf["bb_left"]*1920 
#fdf["bb_top"] = fdf["bb_top"]*1080 
#fdf["bb_width"] = fdf["bb_width"]*1920 
#fdf["bb_height"] = fdf["bb_height"]*1080 
fdf["bb_left"] = fdf["bb_left"] - fdf["bb_width"]/2
fdf["bb_top"] = fdf["bb_top"] - fdf["bb_height"]/2

fdf["bb_left"] = fdf["bb_left"].clip(lower=0, upper=1920)
fdf["bb_top"] = fdf["bb_top"].clip(lower=0, upper=1080)
fdf["bb_width"] = fdf["bb_width"].clip(lower=1, upper=1920)
fdf["bb_height"] = fdf["bb_height"].clip(lower=1, upper=1080)
print(fdf)


path_3 = "results/vigir_submission_1_0.25/"
fdf.to_csv(path_3+"vigir_submission_2.txt", 
                index=False, 
                header=None)

