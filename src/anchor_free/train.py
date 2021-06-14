
# python train.py anchor-free --model-dir ../models/af_mobilenet --splits ../splits/tvsum.yml ../splits/summe.yml --max-epoch 50 --cnn mobilenet --base-model attention --num-feature 1280 --num-head 10 --nms-thresh 0.4 

# python train.py anchor-free --model-dir ../models/af_default --splits ../splits/tvsum.yml ../splits/summe.yml --max-epoch 50 --cnn default --base-model attention --num-feature 1024 --num-head 8 --num-hidden 128 --nms-thresh 0.4 

# python train.py anchor-free --model-dir ../models/af_mobilenet_bilstm --splits ../splits/tvsum.yml ../splits/summe.yml --max-epoch 50 --cnn mobilenet --base-model bilstm --num-feature 1280 --nms-thresh 0.4 

# python train.py anchor-based --model-dir ../models/ab_mobilenet_bilstm --splits ../splits/tvsum.yml ../splits/summe.yml --max-epoch 50 --cnn mobilenet --base-model bilstm --num-feature 1280 --nms-thresh 0.4 

# python train.py anchor-based --model-dir ../models/ab_squeeze --splits ../splits/tvsum.yml ../splits/summe.yml --max-epoch 50 --cnn squeeze --base-model attention --num-feature 1000 --num-head 10 --num-hidden 125 --nms-thresh 0.4 


import logging

import torch
import numpy as np

from anchor_free import anchor_free_helper
from anchor_free.dsnet_af import DSNetAF
from anchor_free.losses import calc_ctr_loss, calc_cls_loss, calc_loc_loss
from evaluate import evaluate
from helpers import data_helper, vsumm_helper
from helpers import init_helper, data_helper, vsumm_helper, bbox_helper


logger = logging.getLogger()


def train(args, split, save_path):
    model = DSNetAF(base_model=args.base_model, num_feature=args.num_feature,
                    num_hidden=args.num_hidden, num_head=args.num_head)
    model = model.to(args.device)

    model.train()

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.lr,
                                 weight_decay=args.weight_decay)

    max_val_fscore = -1

    train_set = data_helper.VideoDataset(split['train_keys'])
    train_loader = data_helper.DataLoader(train_set, shuffle=True)

    val_set = data_helper.VideoDataset(split['test_keys'])
    val_loader = data_helper.DataLoader(val_set, shuffle=False)

    cnn = args.cnn

    for epoch in range(args.max_epoch):
        model.train()
        stats = data_helper.AverageMeter('loss', 'cls_loss', 'loc_loss',
                                         'ctr_loss')

        #for _, seq, gtscore, change_points, n_frames, nfps, picks, _ in train_loader:
        for _, _, n_frames, picks, gtscore, _, _, \
            seq_default, cps_default, nfps_default, \
            seq_lenet, seq_alexnet, seq_mobilenet, seq_squeeze, seq_resnet, \
            seq_lenet_c, seq_alexnet_c, seq_mobilenet_c, seq_squeeze_c, seq_resnet_c, \
            cps_lenet_c, cps_alexnet_c, cps_mobilenet_c, cps_squeeze_c, cps_resnet_c, \
            _, _, _, cps_lenet, cps_alexnet, cps_mobilenet, cps_squeeze, cps_resnet in train_loader:
            
            if cnn == "default":
                seq = seq_default
                cps = cps_default
                nfps = nfps_default
            else: 
                if cnn == "lenet":
                    seq = seq_lenet_c
                    change_points = cps_lenet_c
                if cnn == "alexnet":
                    seq = seq_alexnet_c
                    change_points = cps_alexnet_c
                if cnn == "mobilenet":
                    seq = seq_mobilenet_c
                    change_points = cps_mobilenet_c
                if cnn == "squeeze":
                    seq = seq_squeeze_c
                    change_points = cps_squeeze_c
                if cnn == "resnet":
                    seq = seq_resnet_c
                    change_points = cps_resnet_c

                begin_frames = change_points[:-1]
                end_frames = change_points[1:]
                cps = np.vstack((begin_frames, end_frames)).T
                # Here, the change points are detected (Change-point positions t0, t1, ..., t_{m-1})
                nfps = end_frames - begin_frames

            #seq = seq_resnet
            #cps = cps_default
            #nfps = nfps_default

            keyshot_summ = vsumm_helper.get_keyshot_summ(
                gtscore, cps, n_frames, nfps, picks)
            target = vsumm_helper.downsample_summ(keyshot_summ)

            if not target.any():
                continue

            seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(args.device)

            cls_label = target
            loc_label = anchor_free_helper.get_loc_label(target)
            ctr_label = anchor_free_helper.get_ctr_label(target, loc_label)

            pred_cls, pred_loc, pred_ctr = model(seq)

            cls_label = torch.tensor(cls_label, dtype=torch.float32).to(args.device)
            loc_label = torch.tensor(loc_label, dtype=torch.float32).to(args.device)
            ctr_label = torch.tensor(ctr_label, dtype=torch.float32).to(args.device)

            cls_loss = calc_cls_loss(pred_cls, cls_label, args.cls_loss)
            loc_loss = calc_loc_loss(pred_loc, loc_label, cls_label,
                                     args.reg_loss)
            ctr_loss = calc_ctr_loss(pred_ctr, ctr_label, cls_label)

            loss = cls_loss + args.lambda_reg * loc_loss + args.lambda_ctr * ctr_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            stats.update(loss=loss.item(), cls_loss=cls_loss.item(),
                         loc_loss=loc_loss.item(), ctr_loss=ctr_loss.item())

        # For each epoch, evaluate the model
        args = init_helper.get_arguments()
        seg_algo = args.segment_algo
        cnn = args.cnn

        init_helper.init_logger(args.model_dir, args.log_file)
        init_helper.set_random_seed(args.seed)

        # logger.info(vars(args))

        val_fscore, _ = evaluate(model, cnn, seg_algo, val_loader, args.nms_thresh, args.device)
        # val_fscore, _ = evaluate(model, val_loader, args.nms_thresh, args.device)

        if max_val_fscore < val_fscore:
            max_val_fscore = val_fscore
            torch.save(model.state_dict(), str(save_path))

        logger.info(f'Epoch: {epoch}/{args.max_epoch} '
                    f'Loss: {stats.cls_loss:.4f}/{stats.loc_loss:.4f}/{stats.ctr_loss:.4f}/{stats.loss:.4f} '
                    f'F-score cur/max: {val_fscore:.4f}/{max_val_fscore:.4f}')

    return max_val_fscore
