
## PRE-TRAINED MODELS
# python evaluate2.py anchor-free --model-dir ../models/pretrain_af_basic/ --splits ../splits/tvsum.yml ../splits/summe.yml --nms-thresh 0.4 --segment_algo kts
# python evaluate2.py anchor-based --model-dir ../models/pretrain_ab_basic/ --splits ../splits/tvsum.yml ../splits/summe.yml --segment_algo us

## CUSTOM MODELS
# python evaluate2.py anchor-based --model-dir ../models/ab_tvsum_aug/ --splits ../splits/tvsum_aug.yml --segment_algo us

import logging
from pathlib import Path

import numpy as np
import torch

from helpers import init_helper, data_helper, vsumm_helper, bbox_helper
from modules.model_zoo import get_model

import random

logger = logging.getLogger()

def evaluate(model, seg_algo, val_loader, nms_thresh, device):
    
    model.eval()
    stats = data_helper.AverageMeter('fscore', 'diversity')

    with torch.no_grad():
        # For each video
        for _, test_key, seq, _, change_points_kts, change_points_osg, \
            change_points_osg_sem, change_points_pyths, n_frames, nfps_kts, picks, user_summary in val_loader:
           
            # Model prediction
            seq_torch = torch.from_numpy(seq).unsqueeze(0).to(device)
            pred_cls, pred_bboxes = model.predict(seq_torch)

            # Compress and round all value between 0 and seq_len
            seq_len = len(seq)
            pred_bboxes = np.clip(pred_bboxes, 0, seq_len).round().astype(np.int32)
            #print("pred_boxes: " + str(pred_bboxes[0:5]))
            
            # Apply NMS to pred_cls (condifence scores of segments) and to LR bboxes
            pred_cls, pred_bboxes = bbox_helper.nms(pred_cls, pred_bboxes, nms_thresh)

            # VIDEO SEGMENTATION

            if seg_algo == "kts":
                cps = change_points_kts
                nfps = nfps_kts

            else:

                if seg_algo == "osg":
                    change_points = change_points_osg
                
                if seg_algo == "osg_sem":
                    change_points = change_points_osg_sem

                if seg_algo == "pyths":
                    change_points = change_points_pyths

                if seg_algo == "us":
                    interval = int(n_frames / 20)
                    change_points = np.arange(1, n_frames, interval)
                    change_points = np.hstack((0, change_points, n_frames))

                if seg_algo == "random":
                    change_points = np.sort(random.sample(range(1, n_frames - 1), 20))
                    change_points = np.hstack((0, change_points, n_frames))

                begin_frames = change_points[:-1]
                end_frames = change_points[1:]
                cps = np.vstack((begin_frames, end_frames)).T
                # Here, the change points are detected (Change-point positions t0, t1, ..., t_{m-1})
                nfps = end_frames - begin_frames  # For each segment, calculate the number of frames

            # Convert predicted bounding boxes to summary
            pred_summ = vsumm_helper.bbox2summary(seq_len, pred_cls, pred_bboxes, cps, n_frames, nfps, picks)

            # Compute F-Measure
            eval_metric = 'avg' if 'tvsum' in test_key else 'max'
            fscore = vsumm_helper.get_summ_f1score(pred_summ, user_summary, eval_metric)

            #Down-sample the summary by 15 times
            pred_summ = vsumm_helper.downsample_summ(pred_summ)

            diversity = vsumm_helper.get_summ_diversity(pred_summ, seq)
            # For each video, update the measures
            stats.update(fscore=fscore, diversity=diversity)

            #print(stats.fscore, stats.diversity)

    return stats.fscore, stats.diversity


def main():
    args = init_helper.get_arguments()
    seg_algo = args.segment_algo

    init_helper.init_logger(args.model_dir, args.log_file)
    init_helper.set_random_seed(args.seed)

    logger.info(vars(args))
    model = get_model(args.model, **vars(args))
    model = model.eval().to(args.device)

    for split_path in args.splits:
        split_path = Path(split_path)
        splits = data_helper.load_yaml(split_path)

        stats = data_helper.AverageMeter('fscore', 'diversity')

        # For each split (train/test) in dataset.yml file (x5)
        for split_idx, split in enumerate(splits):
            
            # Load the model from the checkpoint folder
            ckpt_path = data_helper.get_ckpt_path(args.model_dir, split_path, split_idx)
            state_dict = torch.load(str(ckpt_path), map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict)

            # Load split test keys in the VideoDataset Class
            val_set = data_helper.VideoDataset(split['test_keys']) # Inizialization
            val_loader = data_helper.DataLoader(val_set, shuffle=False) # Load data

            # Evaluate the split test keys
            fscore, diversity = evaluate(model, seg_algo, val_loader, args.nms_thresh, args.device)
            stats.update(fscore=fscore, diversity=diversity)

            logger.info(f'{split_path.stem} split {split_idx}: diversity: '
                        f'{diversity:.4f}, F-score: {fscore:.4f}')

        logger.info(f'{split_path.stem}: diversity: {stats.diversity:.4f}, '
                    f'F-score: {stats.fscore:.4f}')


if __name__ == '__main__':
    main()