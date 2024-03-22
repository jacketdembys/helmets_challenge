import numpy as np
import pandas as pd
import sys
from collections import Counter



def correct_class_id(max_ids, class_id_frame):

    class_id_frame_results = class_id_frame.copy()
    rows, cols = class_id_frame_results.shape
    additional_cols = np.zeros((rows, 1))
    class_id_frame_results = np.hstack((class_id_frame_results, additional_cols))

    for i in range(max_ids):
        inds = np.where(class_id_frame[:,0]==i)
        #print(i, class_id_frame[inds[0],1])

        reduced_class_id_frame = class_id_frame[inds[0],:]
        reduced_class_id_frame_results = reduced_class_id_frame.copy()
        counts = Counter(reduced_class_id_frame[:,1])
        most_counted_id = counts.most_common(1)

        rrows, rcols = reduced_class_id_frame_results.shape
        corrected_class = reduced_class_id_frame[:, 1].copy()
        print(corrected_class.shape)
        corrected_class = np.reshape(corrected_class, (rrows,1))
        print(corrected_class.shape)
        

        if len(most_counted_id) > 0:
            most_id = most_counted_id[0][0]
            indices = np.where(reduced_class_id_frame[:, 1] > most_id)[0]
            corrected_class[indices] = most_id
            print()
            print(reduced_class_id_frame)
            reduced_class_id_frame = np.hstack((reduced_class_id_frame, corrected_class))
            print(reduced_class_id_frame)
            reduced_class_id_frame[:, [3,2]] = reduced_class_id_frame[:, [2,3]]
            #print(reduced_class_id_frame)
            #reduced_class_id_frame[indices, 1] = most_id
            #corrected_class[indices] = most_id
            
            #print(corrected_class)
            #print(indices)
            print(reduced_class_id_frame)
            
    
    #print(class_id_frame)

    return reduced_class_id_frame, indices


if __name__ == "__main__":

    class_id_frame = np.load("calss_id_frame.npy")
    print(class_id_frame.shape)
    max_ids = 40

    reduced_class_id_frame, indices = correct_class_id(max_ids, class_id_frame)

    
