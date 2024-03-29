import csv

# Define the number of folders and criteria for each group
num_folders = 10
criteria = [
    {"Fog": 2, "Night": 2, "Day": 6},  # Criteria for folders 1 to 7
    {"Fog": 2, "Night": 1, "Day": 7}   # Criteria for folders 8 to 10
]

# Read video IDs and tags from CSV file
videos = []
with open("video_tags.csv", newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        video_id, tag = row
        videos.append((video_id, tag))

# Group videos based on criteria
folders = []
for i in range(num_folders):
    folder_videos = []
    for tag, count in criteria[i // (num_folders // len(criteria))].items():
        videos_for_tag = [(vid, t) for vid, t in videos if t == tag and (vid, t) not in folder_videos]
        folder_videos.extend(videos_for_tag[:count])
        videos = [(vid, t) for vid, t in videos if (vid, t) not in videos_for_tag[:count]]
    folders.append(folder_videos)

# Distribute remaining videos evenly across folders
if videos:
    for i, (vid, tag) in enumerate(videos):
        folders[i % num_folders].append((vid, tag))

# Print the groups formed with tags
for i, folder in enumerate(folders, start=1):
    print(f"Folder {i}:")
    for video_id, tag in folder:
        print(f"  Video ID: {video_id}, Tag: {tag}")
