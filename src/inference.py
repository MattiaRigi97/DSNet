
## PRE-TRAINED MODELS
# python inference.py anchor-free --model-dir ../models/pretrain_af_basic/ --splits ../splits/tvsum.yml ../splits/summe.yml --nms-thresh 0.4
# python inference.py anchor-based --model-dir ../models/pretrain_ab_basic/ --splits ../splits/tvsum.yml ../splits/summe.yml --nms-thresh 0.4

## CUSTOM MODELS
# python inference.py anchor-based --model-dir ../models/ab_tvsum_aug/ --splits ../splits/tvsum_aug.yml --video_name Fire_Domino.mp4

import logging
from pathlib import Path

import numpy as np
from matplotlib import cm

from helpers import init_helper, data_helper, vsumm_helper, bbox_helper
from helpers.data_helper import open_video
from helpers.data_helper import write_video_from_frame
from helpers.data_helper import scale

from modules.model_zoo import get_model
from kts.cpd_auto import cpd_auto

import torch
from torch import optim, nn
from torchvision import models, transforms

import cv2
from PIL import Image
import imageio

from feature_extraction import FeatureExtractor

logger = logging.getLogger()

def inference(model, feat_extr, frames, n_frame_video, preprocess, nms_thresh, device):
    
    model.eval()

    with torch.no_grad():

        # Extract LeNet Features for all the frames
        #seq = np.asarray([])
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
                output = feat_extr(input_batch)
            #seq.append(np.asarray(output[0].cpu(), dtype=np.float32))
            seq = np.append(seq, output[0].cpu())

        # From 1D Array to 2D array (each of 1024 elements)
        # print("MIN: "+str(seq.min()))
        # print("MAX: "+str(seq.max()))
        seq = np.reshape(seq, (-1,1024))
        seq = np.asarray(seq, dtype=np.float32)
        seq_len = len(seq)
        
        # print("seq: " + str(seq)) 
        # print("seq shape: " + str(seq.shape)+ "\n")
        ## print(type(seq))

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

        # Apply KTS
        #n_frames = seq_len * 15 - 1 
        # print("N_Video Frame: " + str(n_frame_video) + "\n")
        picks = np.arange(0, seq_len) * 15 # array([    0,    15,    30,    45,    60, ...])
        # print("picks: " + str(picks))
        # print("picks shape: " + str(picks.shape) + "\n")

        kernel = np.matmul(seq, seq.T) # Matrix product of two arrays
        kernel = scale(kernel, 0, 1)
        # print("*************\n" + str(kernel))
        # print("*************\n" + str(kernel.shape))
        # print("SEQ LEN: " + str(seq_len))
        change_points, _ = cpd_auto(K = kernel, ncp = seq_len-1, vmax = 1 ) # Call of the KTS Function
        # print("cps: " + str(change_points))
        change_points *= 15
        # print("cps: " + str(change_points))
        change_points = np.hstack((0, change_points, n_frame_video))
        begin_frames = change_points[:-1]
        end_frames = change_points[1:]
        
        change_points = np.vstack((begin_frames, end_frames - 1)).T
        # print("cps: " + str(change_points))
        # print("cps shape: " + str(change_points.shape) + "\n")

        # Here, the change points are detected (Change-point positions t0, t1, ..., t_{m-1})
        n_frame_per_seg = end_frames - begin_frames  # For each segment, calculate the number of frames
        # print("nfps: " + str(n_frame_per_seg))   
        # print("nfps shape: " + str(n_frame_per_seg.shape) + "\n")   

        # Convert predicted bounding boxes to summary
        pred_summ = vsumm_helper.bbox2summary(seq_len, pred_cls, pred_bboxes, change_points, n_frame_video, n_frame_per_seg, picks)
        # print("pred summary: " + str(pred_summ[0:5])) # True, False list
        # print("pred summary len: " + str(sum(pred_summ))) 
        # print("pred summary shape: " + str(pred_summ.shape) + "\n")
    
    return pred_summ


def main():

    # sample execution (requires torchvision)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

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

    # Load video
    video, frames, frames_sel, n_frame_video = open_video(video_name, video_path, sampling_interval=15)

    # Initialize the model
    lenet = models.googlenet(pretrained=True)
    # print(model(torch.randn(1,3,224,224)).shape)
    # print(model)

    # Initialize the feature extractor model
    feat_extr = FeatureExtractor(lenet)
    # print(new_model(torch.randn(1,3,224,224)).shape)
    # print(new_model)

    # Change the device to GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    feat_extr = feat_extr.to(device)

    # Run inference
    pred_summ = inference(model, feat_extr, frames_sel, n_frame_video, preprocess, args.nms_thresh, args.device)
    
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