
## PRE-TRAINED MODELS
# python inference.py anchor-free --model-dir ../models/pretrain_af_basic/ --splits ../splits/tvsum.yml ../splits/summe.yml --cnn default --segment_algo kts --nms-thresh 0.4 --video_name Fire_Domino.mp4
# python inference.py anchor-based --model-dir ../models/pretrain_ab_basic/ --splits ../splits/tvsum.yml ../splits/summe.yml --cnn default --segment_algo kts --video_name Fire_Domino.mp4

## CUSTOM MODELS
# python inference.py anchor-based --model-dir ../models/ab_tvsum_aug/ --splits ../splits/tvsum_aug.yml --cnn default --segment_algo kts --video_name Fire_Domino.mp4

# import cv2
# from matplotlib import cm
# from segmentation.optimal_group.h_nrm import get_optimal_sequence_nrm
# from torch import optim
# import h5py

## PACKAGE
import random
import logging
import numpy as np
from PIL import Image
from pathlib import Path

## SEGMENTATION METHODS
# Segment Detection based on Optimal Grouping
from segmentation.optimal_group.h_add import get_optimal_sequence_add
from segmentation.optimal_group.estimate_scenes_count import estimate_scenes_count
from dictances import bhattacharyya as bt
# Segment Detection based on KTS
from segmentation.kts.cpd_auto import cpd_auto
# Segment Detection with PySceneDetect
from segmentation.pyscenedetecor import find_scenes
from segmentation.pyscenedetecor import mean_pixel_intensity_calc

# Helpers functions
from helpers import init_helper, data_helper, vsumm_helper, bbox_helper
from helpers.data_helper import open_video
from helpers.data_helper import write_video_from_frame
from helpers.data_helper import scale
from modules.model_zoo import get_model

# Features Extraction functions
from feature_extraction import FeatureExtractor
from feature_extraction import generate_bgr_hist
from sklearn import preprocessing

# Torch Modules
import torch
from torch import nn
from torchvision import models, transforms


logger = logging.getLogger()

# Define the preprocessing function for images
preprocess = transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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


def inference(model, feat_extr, shape, filename, frames, n_frame_video, seg_algo, nms_thresh, device):
    
    model.eval()
    feat_extr.eval()

    with torch.no_grad():

        seq = extract_features(feat_extr, frames, shape)

        seq_len = len(seq)

        # print(seq)
        print("MIN: " + str(seq.min()))
        print("MAX: " + str(seq.max()))
        print("MEAN: " + str(seq.mean()))
        print(seq.shape)

        # Model prediction
        seq_torch = torch.from_numpy(seq).unsqueeze(0).to(device)
        pred_cls, pred_bboxes = model.predict(seq_torch)

        # print("OUTPUT")
        # print("******************************************************")
        # print("pred_cls: " + str(pred_cls[30:50]))
        # print("min e max: " + str(pred_cls.min()) + ' ' + str(pred_cls.max()))
        # print("pred_cls shape: " + str(pred_cls.shape) + "\n")
        # print("pred_boxes: " + str(pred_bboxes[30:50]))
        # print("pred_boxes shape: " + str(pred_bboxes.shape) + "\n")

        # Compress and round all value between 0 and seq_len
        pred_bboxes = np.clip(pred_bboxes, 0, seq_len).round().astype(np.int32)
        # print("pred_boxes shape: " + str(pred_bboxes.shape) + "\n")
        # print("pred_boxes: " + str(pred_bboxes[30:50]))
        
        # Apply NMS to pred_cls (condifence scores of segments) and to LR bboxes
        pred_cls, pred_bboxes = bbox_helper.nms(pred_cls, pred_bboxes, nms_thresh)
        # print("\nApply NMS")
        # print("pred_cls shape: " + str(pred_cls.shape))
        # print("pred_boxes: " + str(pred_bboxes[30:50]))
        # print("pred_boxes shape: " + str(pred_bboxes.shape) + "\n")

        # n_frames = seq_len * 15 - 1 
        # print("N_Video Frame: " + str(n_frame_video) + "\n")
        picks = np.arange(0, seq_len) * 15 # array([    0,    15,    30,    45,    60, ...])
        # print("picks: " + str(picks))
        # print("picks shape: " + str(picks.shape) + "\n")

        # VIDEO SEGMENTATION

        if seg_algo == "kts":
            # Apply KTS
            kernel = np.matmul(seq, seq.T) # Matrix product of two arrays
            kernel = scale(kernel, 0, 1)
            change_points, _ = cpd_auto(K = kernel, ncp = seq_len - 1, vmax = 1 ) # Call of the KTS Function
            change_points *= 15
            change_points = np.hstack((0, change_points, n_frame_video)) # add 0 and the last frame
        
        if seg_algo == "osg" or seg_algo == "osg_sem":
            if seg_algo == "osg":
                # Extract color histogram from each frame
                features = generate_bgr_hist(frames, num_bins = 16)
                # Calculate the distance matrix
            else:
                features = seq
            dist_mat = np.zeros(shape=(len(features),len(features)))
            for i in range(0,len(features)):
                for j in range(0,len(features)):
                    di = dict(enumerate(features[i], 1))
                    dj = dict(enumerate(features[j], 1))
                    dist_mat[i,j] = bt(di,dj)
            # Normalize the distance matrix
            dist_mat = (dist_mat - dist_mat.min()) / (dist_mat.max() - dist_mat.min())
            # Estimate the number of scenes/segments
            K = estimate_scenes_count(dist_mat)
            # Find the change points
            change_points = get_optimal_sequence_add(dist_mat, K)
            change_points *= 15
            change_points = np.hstack((0, change_points, n_frame_video)) # add 0 and the last frame

        if seg_algo == "pyths":
            all_mean_value = mean_pixel_intensity_calc(filename)
            print("MEAN: " + str(all_mean_value))
            scenes = find_scenes(filename, th = all_mean_value, min_scene_length = 30, min_perc = 0.8)
            # Select all the split
            change_points = []
            for scene in scenes:
                change_points.append(int(scene[1]))
            change_points = np.hstack((0, change_points)) # append 0 from the change point list

        if seg_algo == "us":
            interval = int(n_frame_video / 20)
            change_points = np.arange(1, n_frame_video, interval)
            change_points = np.hstack((0, change_points, n_frame_video))

        if seg_algo == "random":
            change_points = np.sort(random.sample(range(1, n_frame_video - 1), 20))
            change_points = np.hstack((0, change_points, n_frame_video))

        begin_frames = change_points[:-1]
        end_frames = change_points[1:]
        change_points = np.vstack((begin_frames, end_frames)).T
        # print("cps: " + str(change_points))
        print("cps shape: " + str(change_points.shape) + "\n")

        # Here, the change points are detected (Change-point positions t0, t1, ..., t_{m-1})
        n_frame_per_seg = end_frames - begin_frames  # For each segment, calculate the number of frames
        print("nfps: " + str(n_frame_per_seg))   
        print("nfps shape: " + str(n_frame_per_seg.shape) + "\n")   

        # Convert predicted bounding boxes to summary
        pred_summ = vsumm_helper.bbox2summary(seq_len, pred_cls, pred_bboxes, change_points, n_frame_video, n_frame_per_seg, picks)
        # print("pred summary: " + str(pred_summ[0:5])) # True, False list
        # print("pred summary len: " + str(sum(pred_summ))) 
        # print("pred summary shape: " + str(pred_summ.shape) + "\n")
    
    return pred_summ


def main():

    args = init_helper.get_arguments()

    init_helper.init_logger(args.model_dir, args.log_file)
    init_helper.set_random_seed(args.seed)

    logger.info(vars(args))

    print(vars(args))

    # Load the model
    model = get_model(args.model, **vars(args))
    model = model.eval().to(args.device)

    for split_path in args.splits:
        split_path = Path(split_path)
        splits = data_helper.load_yaml(split_path)

        # For each split (train/test) in dataset.yml file (x5)
        for split_idx, split in enumerate(splits):
            
            # Load the model from the checkpoint folder
            ckpt_path = data_helper.get_ckpt_path(args.model_dir, split_path, split_idx)
            state_dict = torch.load(str(ckpt_path), map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict)

    # Specify video file
    video_path = args.video_path
    video_name = args.video_name
    output_video_path = args.output_path
    filename = video_path + "\\" + video_name
    
    # Define the segmentation algorithm
    seg_algo = args.segment_algo

    # Load video
    video, frames, frames_sel, n_frame_video = open_video(video_name, video_path, sampling_interval=15)

    # Initialize the feature extractor model
    cnn = args.cnn

    if cnn == "lenet" or cnn == "default":
        feat_extr = models.googlenet(pretrained=True)
        feat_extr.eval()
        feat_extr = nn.Sequential(*list(feat_extr.children())[:-2])
        shape = feat_extr(torch.randn(1,3,224,224)).shape[1]
    if cnn == "alexnet":
        feat_extr = models.alexnet(pretrained=True)
        feat_extr.eval()
        new_classifier = nn.Sequential(*list(feat_extr.classifier.children())[:-1])
        feat_extr.classifier = new_classifier
        shape = feat_extr(torch.randn(1,3,224,224)).shape[1]
    if cnn == "mobilenet":
        feat_extr = models.mobilenet_v2(pretrained=True)
        feat_extr.eval()
        new_classifier = nn.Sequential(*list(feat_extr.classifier.children())[:-2])
        feat_extr.classifier = new_classifier
        shape = feat_extr(torch.randn(1,3,224,224)).shape[1]
    if cnn == "squeeze":
        feat_extr = models.squeezenet1_0(pretrained=True)
        feat_extr.eval()
        new_classifier = nn.Sequential(*list(feat_extr.classifier.children())[:4])
        feat_extr.classifier = new_classifier
        shape = feat_extr(torch.randn(1,3,224,224)).shape[1]
    if cnn == "resnet":
        feat_extr = models.resnet18(pretrained=True)
        feat_extr.eval()
        new_classifier = nn.Sequential(*list(feat_extr.children())[:-1])
        feat_extr.classifier = new_classifier
        shape = feat_extr(torch.randn(1,3,224,224)).shape[1]
        
    # Change the device to GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    feat_extr = feat_extr.to(device)

    # Run inference
    pred_summ = inference(model, feat_extr, shape, filename, frames_sel, n_frame_video, seg_algo, args.nms_thresh, args.device)
    # Trasform and save in .mp4 file 
    pred_summ = np.array(pred_summ) # True False Mask
    frames = np.array(frames)
    print("All frames: " + str(frames.shape))

    final_summary = frames[pred_summ,:]
    print("Final summary frames: " + str(final_summary.shape))

    model_name = args.model_dir.split("/")[-2]

    write_video_from_frame(output_video_path, video_name, model_name, final_summary)

if __name__ == '__main__':
    main()