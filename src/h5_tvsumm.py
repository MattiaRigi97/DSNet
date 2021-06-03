import h5py
import mat73
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

import torch
from torchvision import models, transforms
from torch import  nn

from PIL import Image
from sklearn import preprocessing

from segmentation.kts.cpd_auto import cpd_auto

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

# OPEN THE H5 DATA FILE AND ADD VARIABLES

f = h5py.File(filename, 'r+')
video_path = r"C:\Users\matti\OneDrive - Universita degli Studi di Milano-Bicocca\Uni\LM-DataScience\Tesi+Stage\Dataset\TVSum\ydata-tvsum50-v1_1\ydata-tvsum50-v1_1\video"

# IMPORT ALL THE FEATURE EXTRACTOR MODELS

model_lenet = models.googlenet(pretrained=True)
model_lenet.eval()
model_lenet = nn.Sequential(*list(model_lenet.children())[:-2])
shape_lenet = model_lenet(torch.randn(1,3,224,224)).shape[1]

model_alexnet = models.alexnet(pretrained=True)
model_alexnet.eval()
new_classifier = nn.Sequential(*list(model_alexnet.classifier.children())[:-1])
model_alexnet.classifier = new_classifier
shape_alexnet = model_alexnet(torch.randn(1,3,224,224)).shape[1]

model_mobilenet = models.mobilenet_v2(pretrained=True)
model_mobilenet.eval()
new_classifier = nn.Sequential(*list(model_mobilenet.classifier.children())[:-2])
model_mobilenet.classifier = new_classifier
shape_mobilenet = model_mobilenet(torch.randn(1,3,224,224)).shape[1]

model_squeeze = models.squeezenet1_0(pretrained=True)
model_squeeze.eval()
new_classifier = nn.Sequential(*list(model_squeeze.classifier.children())[:4])
model_squeeze.classifier = new_classifier
shape_squeeze = model_squeeze(torch.randn(1,3,224,224)).shape[1]

model_resnet = models.resnet18(pretrained=True)
model_resnet.eval()
new_classifier = nn.Sequential(*list(model_resnet.children())[:-1])
model_resnet.classifier = new_classifier
shape_resnet = model_resnet(torch.randn(1,3,224,224)).shape[1]

# Define the preprocessing function for images
preprocess = transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define a funciton for calculate the features
def extract_features(model, frames, shape):
    with torch.no_grad():
        seq = []
        for frame in frames:
            im = Image.fromarray(frame)
            input_tensor = preprocess(im)
            input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
            # move the input and model to GPU for speed if available
            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                model.to('cuda')
            with torch.no_grad():
                output = model(input_batch)
            seq = np.append(seq, np.array(output[0].cpu()))
        seq = np.reshape(seq, (-1, shape))
        seq = np.asarray(seq, np.float32)
        seq = preprocessing.normalize(seq)
        return seq

# Define functions for KTS
def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 

def find_cps(seq):
    kernel = np.matmul(seq, seq.T) # Matrix product of two arrays
    kernel = scale(kernel, 0, 1)
    change_points, _ = cpd_auto(K = kernel, ncp = len(seq) - 1, vmax = 1 )
    change_points *= 15
    change_points = np.hstack((0, change_points, n_frame_video))
    return change_points

i = 0
for name in video_names:
    i += 1
    # Save the filename
    video_name = name[1] + str(".mp4")
    """
    filename = video_path + "\\" + str(video_name)
    f.create_dataset(name[0] + '/filename', data=filename)
    
    ### CPS FOR OSG
    # Create the histogram of colors
    video, frames, frames_sel, n_frame_video = open_video(video_name, video_path, sampling_interval = 60)
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
    print("*** " + str(K))
    change_points = get_optimal_sequence_add(dist_mat, K)
    change_points *= 15
    change_points = np.hstack((0, change_points, n_frame_video)) # add 0 and the last frame
    f.create_dataset(name[0] + '/change_points_osg', data = change_points)
    
    ### CPS FOR OSG_SEM
    seq = np.asarray(f[name[0]]["features"][::4])
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
    print("*** " + str(K))
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

    print("**** " + str(num) + " Video Processed ****")"""

    video, frames, frames_sel, n_frame_video = open_video(video_name, video_path, sampling_interval=15)

    seq_lenet = extract_features(model_lenet, frames_sel, shape_lenet)
    seq_alexnet = extract_features(model_alexnet, frames_sel, shape_alexnet)
    seq_mobilenet = extract_features(model_mobilenet, frames_sel, shape_mobilenet)
    seq_squeeze = extract_features(model_squeeze, frames_sel, shape_squeeze)
    seq_resnet = extract_features(model_resnet, frames_sel, shape_resnet)

    f.create_dataset(name[0] + '/seq_lenet', data = seq_lenet)
    f.create_dataset(name[0] + '/seq_alexnet', data = seq_alexnet)
    f.create_dataset(name[0] + '/seq_mobilenet', data = seq_mobilenet)
    f.create_dataset(name[0] + '/seq_squeeze', data = seq_squeeze)
    f.create_dataset(name[0] + '/seq_resnet', data = seq_resnet)

    cps_lenet = find_cps(seq_lenet)
    cps_alexnet = find_cps(seq_alexnet)
    cps_mobilenet= find_cps(seq_mobilenet)
    cps_squeeze = find_cps(seq_squeeze)
    cps_resnet = find_cps(seq_resnet)

    f.create_dataset(name[0] + '/cps_lenet', data = cps_lenet)
    f.create_dataset(name[0] + '/cps_alexnet', data = cps_alexnet)
    f.create_dataset(name[0] + '/cps_mobilenet', data = cps_mobilenet)
    f.create_dataset(name[0] + '/cps_squeeze', data = cps_squeeze)
    f.create_dataset(name[0] + '/cps_resnet', data = cps_resnet)

    print("**** " + str(i) + " Video Processed ****")

f.close()