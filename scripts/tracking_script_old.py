from collections import defaultdict
import cv2
import numpy as np
from ultralytics import trackers
from ultralytics import YOLO
import pandas as pd
import sys
from collections import Counter

def convert_bbox_to_corners(bbox):
    cx, cy, w, h = bbox
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return [x1, y1, x2, y2]


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def iou_condition_met(track_id_i, track_id_j, frame):
    i_track_id = track_id_i[0]
    i_box = track_id_i[1]
    i_class = track_id_i[2]

    j_track_id = track_id_j[0]
    j_box = track_id_j[1]
    j_class = track_id_j[2]

    score = iou(convert_bbox_to_corners(i_box),convert_bbox_to_corners(j_box))

    if score > 0.20:
        return True 
    else:
        return False


def yolo_track(model, video_path):

    trackers.basetrack.BaseTrack.reset_id()
    results = model.track(source=video_path, conf=0.3, iou=0.5, show=False, verbose=False)

    return results


def build_iou_array(results, num_frames = 200):

    max_tracks = trackers.basetrack.BaseTrack.next_id() - 1 # This should be set to the maximum number of track IDs
    # max_tracks = 31
    num_frames = len(results)

    # # Initialize the 3D array
    iou_conditions_met = np.zeros((num_frames, max_tracks+1, max_tracks+1), dtype=int)
    track_ids_per_frame = {}
    # # Example track IDs per frame (replace with your actual method of obtaining them)
    for frame in range(num_frames):
    #for frame in range(len(results)):
        if results[frame].boxes.id is not None:
            track_ids_per_frame.update({frame: results[frame].boxes.id.int().cpu().tolist()})
        else:
            track_ids_per_frame.update({frame: []})
            
    # # Populate the array
    for frame in range(num_frames):
    #for frame in range(len(results)):
        frame_classes = results[frame].boxes.cls.cpu().tolist()
        frame_boxes = results[frame].boxes.xywh.cpu().tolist()
        track_ids = track_ids_per_frame[frame]
        for i in range(len(track_ids)):
            for j in range(len(track_ids)):
                if iou_condition_met([track_ids[i], frame_boxes[i], frame_classes[i]], [track_ids[j], frame_boxes[j], frame_classes[j]], frame):
                    iou_conditions_met[frame, track_ids[i], track_ids[j]] = 1
    
    return iou_conditions_met


def identify_group(built_array):
    all_groups = {}
    for frame in range(built_array.shape[0]):

        test_frame = built_array[frame].squeeze()

        groups = []
        
        ids = test_frame
        r, c = np.shape(ids)
        for i in range(r):
            idx = np.where(ids[i, :] == 1)[0]
            idx = list(idx)
            if len(idx) > 0:
                if set(idx) in groups:
                    continue
                else:
                    groups.append(set(idx))
        # print("Frame = ", frame," groups = ", groups)
        all_groups.update({frame: groups})
    
    return all_groups


def class_id_frame_map(results, num_frames):

    class_id_assocciation = []
    num_frames = len(results)
    
    for frame in range(num_frames):
        if results[frame].boxes.id is not None:
            ids = results[frame].boxes.id.int().cpu().tolist()
            cls = results[frame].boxes.cls.int().cpu().tolist()
    
            for i in range(len(ids)):
                class_id_assocciation.append([ids[i], cls[i], frame])
    class_id_frame_map_array = np.array(class_id_assocciation)

    return class_id_frame_map_array



def valid_group_identification(all_groups):
    all_group_list = []
    
    for frame in all_groups:
        # print(frame_group)
        for s in all_groups[frame]:
            # print(s)
            all_group_list.append(frozenset(s))
    
    frequencies = Counter(all_group_list)
    union_of_all_sets = set().union(*all_group_list)
    
    intersections_with_frequencies = {element: {} for element in union_of_all_sets}
    
    for element in union_of_all_sets:
        for group in all_group_list:
            intersection = group.intersection({element})
            if intersection:
                # If there is an intersection, get the frequency of this group
                group_as_set = set(group)  # Convert frozenset back to set for reporting
                frequency = frequencies[group]
                intersections_with_frequencies[element][str(group_as_set)] = frequency
    
    max_frequency_groups = {}

    for element, groups in intersections_with_frequencies.items():
        # Initialize variables to keep track of the max group and its frequency
        max_group = None
        max_frequency = -1
        max_group_size = 0  # Keep track of the size of the max group

        # Also track the most frequent group that has more than one member
        most_frequent_multi_member_group = None
        most_frequent_multi_member_group_frequency = -1

        for group_str, frequency in groups.items():
            group_set = eval(group_str)  # Convert the string representation back to a set
            group_size = len(group_set)

            # Check if this group has the highest frequency seen so far
            if frequency > max_frequency:
                max_frequency = frequency
                max_group = group_str
                max_group_size = group_size

            # Independently check if this is the most frequent multi-member group
            if group_size > 1 and frequency > most_frequent_multi_member_group_frequency:
                most_frequent_multi_member_group_frequency = frequency
                most_frequent_multi_member_group = group_str

        # Determine which group to return for this element
        if max_group_size > 1:
            max_frequency_groups[element] = max_group
        else:
            if most_frequent_multi_member_group is not None:
                max_frequency_groups[element] = most_frequent_multi_member_group
            else:
                max_frequency_groups[element] = max_group
    
    

    return max_frequency_groups, intersections_with_frequencies



def find_missing_pair_member(max_frequency_groups, intersections_with_frequencies):

    missing_members = {}

    # Convert the string representation back to sets for valid groups
    valid_groups = {element: eval(group_str) for element, group_str in max_frequency_groups.items()}

    # Iterate over all the groups
    for element, groups in intersections_with_frequencies.items():
        for group_str, frequency in groups.items():
            group_set = eval(group_str)  # Convert the string representation back to a set

            # We're only interested in non-valid, single-member groups.
            if len(group_set) == 1:
                # Find the valid group associated with this single member
                member = next(iter(group_set))  # Get the single member
                valid_group = valid_groups.get(member)

                if valid_group:
                    # Find the missing members by subtracting the single member from the valid group
                    missing_members_in_group = valid_group - group_set

                    if missing_members_in_group:
                        # If there are missing members, record them
                        missing_members[group_str] = missing_members_in_group

    return missing_members

def find_frame_id_for_missing_pairs(missing_members, all_groups):

    frame_ids_for_missing_members = {str(missing): [] for missing in missing_members.values()}

    # Loop over each frame and its groups
    for frame_id, groups in all_groups.items():
        for group in groups:
            # Convert the group to a string for comparison
            group_str = str(group)
            
            # Check if the group is a key in 'missing_members'
            if group_str in missing_members:
                # If so, add the frame ID to the list for the corresponding missing member(s)
                missing = missing_members[group_str]
                missing_str = str(missing)
                frame_ids_for_missing_members[missing_str].append(frame_id)

    return frame_ids_for_missing_members


def find_frame_and_valid_group_for_missing_pairs(all_groups, missing_members, max_frequency_groups):

    valid_groups = max_frequency_groups

    frame_ids_with_valid_group_for_missing = {str(missing): [] for missing in missing_members.values()}

    # Loop over each frame and its groups
    for frame_id, groups in all_groups.items():
        for group in groups:
            # Convert the group to a string for comparison
            group_str = str(group)
            
            # Check if the group is a key in 'missing_members'
            if group_str in missing_members:
                # Find the missing members
                missing = missing_members[group_str]
                for missing_member in missing:
                    # Retrieve the valid group for the missing member
                    valid_group_for_missing = valid_groups.get(missing_member)
                    
                    # If there's a valid group for the missing member, record it along with the frame ID
                    if valid_group_for_missing:
                        frame_ids_with_valid_group_for_missing[str(missing)].append({
                            "frame_id": frame_id,
                            "valid_group": valid_group_for_missing
                        })

    return frame_ids_with_valid_group_for_missing


def track_id_window_extractor(results, track_id, frame_number):

    frame_ids = results[frame_number].boxes.id.int().cpu().numpy()
    frame_cls = results[frame_number].boxes.cls.int().cpu().numpy()
    frame_boxes = results[frame_number].boxes.xywh.int().cpu().numpy()

    id_cls = frame_cls[np.argwhere(frame_ids==track_id)]
    id_box = frame_boxes[np.argwhere(frame_ids==track_id)]

    return id_cls.squeeze(), id_box.squeeze()


def interpolation_known_points_extractor(results, track_id, frame_numbers):

    classes = []
    boxes = []
    for frame in frame_numbers:
        id_cls, id_box = track_id_window_extractor(results, track_id, frame)
        classes.append(id_cls)
        boxes.append(id_box)
    return classes, boxes

def find_closest_frames_for_missing_id(all_groups, 
                                        max_frequency_groups,
                                        frame_ids_with_valid_group_for_missing):
    valid_groups = max_frequency_groups
    # Assuming 'frame_dict' is your initial dictionary with frame IDs as keys and lists of groups as values,
    # and 'valid_groups' is defined as shown previously.

    # Convert 'frame_dict' to have frozensets for easier comparison
    frame_dict_frozensets = {frame_id: [frozenset(group) for group in groups] for frame_id, groups in all_groups.items()}

    valid_group_to_frames = {str(group): [] for group in valid_groups.values()}
    for frame_id, groups in all_groups.items():
        for group in groups:
            group_str = str(group)
            if group_str in valid_group_to_frames:
                valid_group_to_frames[group_str].append(frame_id)

    # Convert lists to sorted sets for efficient lookup
    for group_str in valid_group_to_frames.keys():
        valid_group_to_frames[group_str] = sorted(set(valid_group_to_frames[group_str]))

    # Now, for each missing element, find the two closest frames with the valid group.
    closest_frames_for_missing = {}
    for missing_str, frame_infos in frame_ids_with_valid_group_for_missing.items():
        for info in frame_infos:
            frame_id = info['frame_id']
            valid_group_str = str(info['valid_group'])

            # Get all frames where the valid group appears
            valid_frames = valid_group_to_frames[valid_group_str]

            # Find the two closest frames (before or after) the current frame where the missing element is detected
            diffs = [(abs(frame_id - vf), vf) for vf in valid_frames]
            closest_two = sorted(diffs, key=lambda x: x[0])[:2]
            closest_frame_ids = [frame_id for _, frame_id in closest_two]

            # Store the results
            if missing_str not in closest_frames_for_missing:
                closest_frames_for_missing[missing_str] = []
            closest_frames_for_missing[missing_str].append({
                "missing_frame": frame_id,
                "closest_frames": closest_frame_ids
            })

    closest_frames_for_missing_with_completing_elements = {}

    for missing_str, frame_infos in frame_ids_with_valid_group_for_missing.items():
        missing_elements = eval(missing_str)  # Convert string back to a set
        for info in frame_infos:
            frame_id = info['frame_id']
            valid_group = eval(info['valid_group'])
            valid_group_str = str(valid_group)

            # Get all frames where the valid group appears
            valid_frames = valid_group_to_frames[valid_group_str]

            # Find the two closest frames (before or after) the current frame where the missing element is detected
            diffs = [(abs(frame_id - vf), vf) for vf in valid_frames]
            closest_two = sorted(diffs, key=lambda x: x[0])[:2]
            closest_frame_ids = [frame_id for _, frame_id in closest_two]

            # Find the elements that complete the missing element for valid groups
            completing_elements = valid_group - missing_elements

            # Store the results, including the completing elements
            if missing_str not in closest_frames_for_missing_with_completing_elements:
                closest_frames_for_missing_with_completing_elements[missing_str] = []
            closest_frames_for_missing_with_completing_elements[missing_str].append({
                "missing_frame": frame_id,
                "closest_frames": closest_frame_ids,
                "completing_elements": completing_elements
            })

    

    return closest_frames_for_missing_with_completing_elements


def correct_class_id(class_id_frame, threshold=0.9):

    class_id_frame_results = []
    all_ids = Counter(class_id_frame[:,0])
    all_ids = sorted(all_ids)

    for i in all_ids:
        inds = np.where(class_id_frame[:,0]==i)

        reduced_class_id_frame = class_id_frame[inds[0],:]
        counts = Counter(reduced_class_id_frame[:,1])
        most_counted_id = counts.most_common(1)              

        if len(most_counted_id) > 0:
            rrows, rcols = reduced_class_id_frame.shape
            corrected_class = reduced_class_id_frame[:, 1].copy()
            corrected_class = np.reshape(corrected_class, (rrows,1))
            most_id = most_counted_id[0][0]

            # If most_counted_id appears more than the threshold
            # print(i, most_counted_id[0][1]/rrows)
            if most_counted_id[0][1]/rrows >= threshold:           
                indices = np.where(reduced_class_id_frame[:, 1] > most_id)[0]
                corrected_class[indices] = most_id
                reduced_class_id_frame = np.hstack((reduced_class_id_frame, corrected_class))
                reduced_class_id_frame[:, [3,2]] = reduced_class_id_frame[:, [2,3]]
            else:
                reduced_class_id_frame = np.hstack((reduced_class_id_frame, corrected_class))
                reduced_class_id_frame[:, [3,2]] = reduced_class_id_frame[:, [2,3]]

            class_id_frame_results.append(pd.DataFrame(reduced_class_id_frame, columns=None))

    return np.array(pd.concat(class_id_frame_results))


def triangular_interpolation(red_boxes, blue_boxes):

    red_box_1 = red_boxes[0]
    red_box_2 = red_boxes[1]
    red_box_3 = red_boxes[2]
    
    blue_box_1 = blue_boxes[0]
    blue_box_2 = blue_boxes[1]

    x_red_1 = red_box_1[0]
    y_red_1 = red_box_1[1]
    
    x_red_2 = red_box_2[0]
    y_red_2 = red_box_2[1]
    
    x_red_3 = red_box_3[0]
    y_red_3 = red_box_3[1]
    
    alpha1 = np.arctan((y_red_2 - y_red_1)/(x_red_2 - x_red_1 + 0.00001))
    r1_mag = np.sqrt((y_red_2 - y_red_1)**2 + (x_red_2 - x_red_1)**2)
    
    alpha2 = np.arctan((y_red_3 - y_red_2)/(x_red_3 - x_red_2 + 0.00001))
    r2_mag = np.sqrt((y_red_3 - y_red_2)**2 + (x_red_3 - x_red_2)**2)
    
    alpha_g = alpha2 - alpha1

    
    x_blue_1 = blue_box_1[0]
    y_blue_1 = blue_box_1[1]
    
    x_blue_2 = blue_box_2[0]
    y_blue_2 = blue_box_2[1]
    
    beta_1 = np.arctan((y_blue_2 - y_blue_1)/(x_blue_2 - x_blue_1 + 0.00001))
    b1_mag = np.sqrt((y_blue_2 - y_blue_1)**2 + (x_blue_2 - x_blue_1)**2)
    
    beta2 = beta_1 + alpha_g
    b2_mag = (r2_mag * b1_mag)/r1_mag
    
    x_blue_3 = x_blue_2 - b2_mag * np.cos(beta2)
    y_blue_3 = y_blue_2 - b2_mag * np.sin(beta2)

    w_red_2 = red_box_2[2]
    h_red_2 = red_box_2[3]
    
    w_red_3 = red_box_3[2]
    h_red_3 = red_box_3[3]
    
    w_blue_2 = blue_box_2[2]
    h_blue_2 = blue_box_2[3]
    
    w_blue_3 = (w_red_3 * w_blue_2)/w_red_2
    h_blue_3 = (h_red_3 * h_blue_2)/h_red_2
    
    return [x_blue_3, y_blue_3, w_blue_3, h_blue_3]

def track_id_to_class_id(results, track_id, nym_frames=200):
    all_track_ids = []
    all_class_ids = []

    for frame in range(nym_frames):
        working_results = results[frame]
        try:
            all_track_ids.extend(working_results.boxes.id.cpu().tolist())
            all_class_ids.extend(working_results.boxes.cls.cpu().tolist())
        except:
            continue

    all_track_ids = np.array(all_track_ids)
    all_class_ids = np.array(all_class_ids)

    des_index = np.where(all_track_ids == track_id)
    class_id = all_class_ids[des_index[0][0]]
    
    return class_id

def parse_text_file_corrected(file_path):
    """
    Correctly parses a text file to extract class IDs, box coordinates, and confidence scores.
    
    :param file_path: The path to the text file.
    :return: A tuple of lists containing class IDs, box coordinates, and confidence scores.
    """
    class_ids = []
    boxes = []
    conf_scores = []
    track_ids = []

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
            if not lines:
                print("The file is empty or could not be read.")
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 7:
                    print(f"Skipping line due to incorrect format: {line.strip()}")
                    continue  # Skip lines that don't have exactly 7 parts
                
                # Adjusting based on the new understanding of the format
                box = list(map(float, parts[:5]))  # First 5 parts are the box coordinates
                conf_score = float(parts[5])  # Second to last part is the confidence score
                class_id = int(parts[0])  # Last part is the class ID
                track_id = int(parts[6])

                class_ids.append(class_id)
                boxes.append(box)
                conf_scores.append(conf_score)
                track_ids.append(track_id)
    except Exception as e:
        print(f"An error occurred: {e}")

    return class_ids, boxes, conf_scores, track_ids

def change_number_in_file(file_path, line_number,new_value, number_position=1):
    """
    Changes a specific number in a text file.

    :param file_path: The path to the text file.
    :param line_number: The line number to change (1-indexed).
    :param number_position: The position of the number in the line to change (1-indexed).
    :param new_value: The new value for the number.
    """
    try:
        # Read all lines from the file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Check if the specified line number is within the range of the file's lines
        if line_number > len(lines) or line_number < 1:
            print("Error: The specified line number is out of the file's range.")
            return
        
        # Select the specific line
        line = lines[line_number - 1]
        parts = line.strip().split()
        
        # Check if the specified number position is within the range of the line's parts
        if number_position > len(parts) or number_position < 1:
            print("Error: The specified number position is out of the line's range.")
            return
        
        # Change the specified number
        parts[number_position - 1] = str(new_value)
        # Reconstruct the line and update it in the lines list
        lines[line_number - 1] = ' '.join(parts) + '\n'
        
        # Write the modified lines back to the file
        with open(file_path, 'w') as file:
            file.writelines(lines)
        
        print("File updated successfully.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

def correct_post_1(text_files_path, video_id, class_id_frame_corrections):

    for i in range(class_id_frame_corrections.shape[0]):
        working_list = class_id_frame_corrections[i]
        if working_list[1] != working_list[2]:
            print("correction found!")
            frame_id = working_list[3]
            track_id = working_list[0]
            old_class = working_list[1]
            new_class = working_list[2]
    
            text_file_path = f"{text_files_path}/{video_id}_{frame_id+1}.txt"
            txt_class_ids, txt_boxes, txt_conf_scores, txt_track_ids = parse_text_file_corrected(text_file_path)
            line_position = np.where(txt_track_ids == track_id)[0][0]
            change_number_in_file(text_file_path, line_position + 1,new_class, number_position=1)
            print("Frame = ", frame_id)


def insert_line_into_file(file_path, class_id, box_values, conf_score, track_id, line_number=None):
    """
    Inserts a new line into the text file with the specified class ID, box values, and confidence score.

    :param file_path: The path to the text file.
    :param class_id: The class ID as an integer.
    :param box_values: The box values as a list of four numbers (coordinates).
    :param conf_score: The confidence score as a float.
    :param line_number: Optional; the line number at which to insert the new line (1-indexed). If not specified or
                        if the line number is greater than the file length, the new line will be appended at the end.
    """
    new_line = f"{class_id} {' '.join(map(str, box_values))} {conf_score} {track_id}\n"
    
    try:
        # Read all lines from the file
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Determine where to insert the new line
        if line_number is not None and 1 <= line_number <= len(lines) + 1:
            lines.insert(line_number - 1, new_line)
        else:
            # Append at the end if no valid line number is provided or it's beyond the file length
            lines.append(new_line)
        
        # Write the lines back to the file
        with open(file_path, 'w') as file:
            file.writelines(lines)
        
        print("New line inserted successfully.")
        
    except Exception as e:
        print(f"An error occurred: {e}")


def line_insert_post_2(results, text_files_path, video_id, closest_frames_for_missing_with_completing_elements):
    
    for element in closest_frames_for_missing_with_completing_elements:
    
        red_boxes = []
        blue_boxes = []
        # (num,) = eval(element)
        # print(element)
        
        working_list = closest_frames_for_missing_with_completing_elements[element]
        for correction_list in working_list:
            print("Corerection_list = ", correction_list)
        
            try:
                missing = correction_list['missing_frame']
                closest_frames = correction_list['closest_frames']
                completing_elements = int(list(correction_list['completing_elements'])[0])
                print("missing_frame = ", missing)
                red_frames = closest_frames.copy()
                red_frames.append(missing)
            
                blue_frames = closest_frames.copy()
            
                red_cls, red_boxes = interpolation_known_points_extractor(results, completing_elements, red_frames)
                blue_cls, blue_boxes = interpolation_known_points_extractor(results, int(list(eval(element))[0]), blue_frames)
                # print(working_list)
            
    
    
                interpolated_array = triangular_interpolation(red_boxes, blue_boxes)
                print("successful!")
                print(red_frames) 
                print("Frame = ", missing, "Id = ", element, "intep = ", interpolated_array)
    
                text_file_path = f"{text_files_path}/{video_id}_{missing+1}.txt"
                missing_class = track_id_to_class_id(results,int(list(eval(element))[0]), nym_frames=200)
    
                insert_box = [interpolated_array[0]/1920, interpolated_array[1]/1080, interpolated_array[2]/1920, interpolated_array[3]/1080]
                insert_line_into_file(text_file_path, missing_class, insert_box, 1.0, int(list(eval(element))[0]), line_number=None) 
                
                # image = cv2.imread(f'/home/macula/SMATousi/ai_city/helmet/Deep-SORT-PyTorch/track18/045_training_frames//{missing + 1}.jpg')  
                # box = interpolated_array
    
                
                # image_with_box = draw_box(image.copy(), box, title=f"{int(list(eval(element))[0]), missing_class}")
                # image_with_box = cv2.resize(image_with_box, (960, 540))
                # # Display the image with the box
                # cv2.imshow('Image with Box', image_with_box)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
            except Exception as error:
                print("Error occured!: ", error)
                continue
