import h5py
import numpy as np
from dictances import bhattacharyya as bt
# Features Extraction functions
from feature_extraction import generate_bgr_hist
from helpers.data_helper import open_video
# Segment Detection with PySceneDetect
from segmentation.pyscenedetecor import mean_pixel_intensity_calc
from segmentation.optimal_group.h_add import get_optimal_sequence_add
# Segment Detection based on Optimal Grouping
from segmentation.optimal_group.estimate_scenes_count import estimate_scenes_count
# Segment Detection with PySceneDetect
from segmentation.pyscenedetecor import find_scenes
from segmentation.pyscenedetecor import mean_pixel_intensity_calc

# RETRIEVE THE video_i | video_name relationship for SUMME Dataset
filename = r'C:\Users\matti\github\DSNet\datasets\eccv16_dataset_summe_google_pool5.h5'
with h5py.File(filename, "r") as f:
    video_names = []
    for i in range(1, len(f) + 1):
        video_name = str("video_") + str(i)
        video_names.append([video_name, f[video_name]["video_name"][()].decode("utf-8")])

# video_names[i] = name = [video_1, "Air_Force_One"]
print(video_names)

# OPEN THE H5 DATA FILE AND ADD VARIABLES

f = h5py.File(filename, 'r+')
video_path = r"C:\Users\matti\OneDrive - Universita degli Studi di Milano-Bicocca\Uni\LM-DataScience\Tesi+Stage\Dataset\SumMe\SumMe\videos"

for name in video_names:
    
    #Save the filename
    video_name = name[1] + str(".mp4")
    filename = video_path + "\\" + str(video_name)
    f.create_dataset(name[0] + '/filename', data=filename)
    
    ### CPS FOR OSG
    # Create the histogram of colors
    video, frames, frames_sel, n_frame_video = open_video(video_name, video_path, sampling_interval=15)
    hist = generate_bgr_hist(frames_sel, num_bins = 16)
    f.create_dataset(name[0] + '/rgb_hist', data=hist)
    # Compute the distance matrix
    dist_mat = np.zeros(shape=(len(hist),len(hist)))
    for i in range(0,len(hist)):
        for j in range(0,len(hist)):
            di = dict(enumerate(hist[i], 1))
            dj = dict(enumerate(hist[j], 1))
            dist_mat[i,j] = bt(di,dj)
    dist_mat = (dist_mat - dist_mat.min()) / (dist_mat.max() - dist_mat.min())
    f.create_dataset(name[0] + '/rgb_dist_mat', data=dist_mat)
    # Find the change points
    K = estimate_scenes_count(dist_mat)
    change_points = get_optimal_sequence_add(dist_mat, K)
    change_points *= 15
    change_points = np.hstack((0, change_points, n_frame_video)) # add 0 and the last frame
    f.create_dataset(name[0] + '/change_points_osg', data = change_points)
    
    ### CPS FOR OSG_SEM
    seq = np.asarray(f[name[0]]["features"])
    # Compute the distance matrix between CNN features
    dist_mat = np.zeros(shape=(len(seq),len(seq)))
    for i in range(0,len(seq)):
        for j in range(0,len(seq)):
            di = dict(enumerate(seq[i], 1))
            dj = dict(enumerate(seq[j], 1))
            dist_mat[i,j] = bt(di,dj)
    dist_mat = (dist_mat - dist_mat.min()) / (dist_mat.max() - dist_mat.min())
    f.create_dataset(name[0] + '/seq_dist_mat', data = dist_mat)
    # Find the change points
    K = estimate_scenes_count(dist_mat)
    change_points = get_optimal_sequence_add(dist_mat, K)
    change_points *= 15
    change_points = np.hstack((0, change_points, n_frame_video)) # add 0 and the last frame
    f.create_dataset(name[0] + '/change_points_osg_sem', data = change_points)

    ### CPS FOR PYTHS
    all_mean_value = mean_pixel_intensity_calc(filename)
    scenes = find_scenes(filename, th = all_mean_value, min_scene_length = 30, min_perc = 0.8)
    # Select all the split
    change_points = []
    for scene in scenes:
        change_points.append(int(scene[1]))
    change_points = np.hstack((0, change_points)) # append 0 from the change point list
    f.create_dataset(name[0] + '/change_points_pyths', data = change_points)

    print("**** Video Processed ****")

f.close()