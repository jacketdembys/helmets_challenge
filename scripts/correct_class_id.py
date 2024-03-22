import numpy as np
import pandas as pd
import sys
from collections import Counter



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
            print(i, most_counted_id[0][1]/rrows)
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


if __name__ == "__main__":

    class_id_frame = np.load("calss_id_frame.npy")
    print(class_id_frame)

    class_id_frame_results = correct_class_id(class_id_frame)
    np.set_printoptions(threshold=sys.maxsize)
    print(class_id_frame_results)

    
