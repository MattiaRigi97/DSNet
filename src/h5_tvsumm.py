import h5py
import mat73
import numpy as np
from dictances import bhattacharyya as bt
# Features Extraction functions
from feature_extraction import generate_bgr_hist
from helpers.data_helper import open_video
# Segment Detection with PySceneDetect
from segmentation.pyscenedetecor import mean_pixel_intensity_calc

# RETRIEVE THE video_i | video_name relationship for TVSUMM Dataset
filename = r'C:\Users\matti\github\DSNet\datasets\eccv16_dataset_tvsum_google_pool5.h5'
gt_data = r'C:\Users\matti\OneDrive - Universita degli Studi di Milano-Bicocca\Uni\LM-DataScience\Tesi+Stage\Dataset\TVSum\ydata-tvsum50-v1_1\ydata-tvsum50-v1_1\matlab\ydata-tvsum50.mat'
data_dict = mat73.loadmat(gt_data)
name_list = data_dict["tvsum50"]["video"]
video_names = []
for i in range(0, len(name_list)):
        video_name = str("video_") + str(i + 1)
        video_names.append([video_name, name_list[i]])

# video_names[i] = name = [video_1, "Air_Force_One"]
print(video_names)

f = h5py.File(filename, 'r+')
video_path = r"C:\Users\matti\OneDrive - Universita degli Studi di Milano-Bicocca\Uni\LM-DataScience\Tesi+Stage\Dataset\TVSum\ydata-tvsum50-v1_1\ydata-tvsum50-v1_1\video"
for name in video_names:
    #Save the filename
    video_name = name[1] + str(".mp4")
    filename = video_path + "\\" + str(video_name)
    f.create_dataset(name[0] + '/filename', data=filename)
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
    # Compute the all mean intensity value
    all_mean_value = mean_pixel_intensity_calc(filename)
    f.create_dataset(name[0] + '/all_mean_intensity', data=all_mean_value)
f.close()