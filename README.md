# DSNet: A Flexible Detect-to-Summarize Network for Video Summarization [[paper]](https://ieeexplore.ieee.org/document/9275314)

[![UnitTest](https://github.com/li-plus/DSNet/workflows/UnitTest/badge.svg)](https://github.com/li-plus/DSNet/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/li-plus/DSNet/blob/main/LICENSE)

![framework](docs/framework.jpg)

A PyTorch implementation of our paper [DSNet: A Flexible Detect-to-Summarize Network for Video Summarization](https://ieeexplore.ieee.org/document/9275314) by [Wencheng Zhu](https://woshiwencheng.github.io/), [Jiwen Lu](http://ivg.au.tsinghua.edu.cn/Jiwen_Lu/), [Jiahao Li](https://github.com/li-plus), and [Jie Zhou](http://www.au.tsinghua.edu.cn/info/1078/1635.htm). Published in [IEEE Transactions on Image Processing](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=83).

## Getting Started

This project is developed on Ubuntu 16.04 with CUDA 9.0.176.

First, clone this project to your local environment.

```sh
git clone https://github.com/li-plus/DSNet.git
```

Create a virtual environment with python 3.6, preferably using [Anaconda](https://www.anaconda.com/).

```sh
conda create --name dsnet python=3.6
conda activate dsnet
```

Install python dependencies.

```sh
pip install -r requirements.txt
```

## Datasets Preparation

Download the pre-processed datasets into `datasets/` folder, including [TVSum](https://github.com/yalesong/tvsum), [SumMe](https://gyglim.github.io/me/vsum/index.html), [OVP](https://sites.google.com/site/vsummsite/download), and [YouTube](https://sites.google.com/site/vsummsite/download) datasets.

```sh
mkdir -p datasets/ && cd datasets/
wget https://www.dropbox.com/s/tdknvkpz1jp6iuz/dsnet_datasets.zip
unzip dsnet_datasets.zip
```

If the Dropbox link is unavailable to you, try downloading from below links.

+ (Baidu Cloud) Link: <https://pan.baidu.com/s/1LUK2aZzLvgNwbK07BUAQRQ> Extraction Code: x09b
+ (Google Drive) <https://drive.google.com/file/d/11ulsvk1MZI7iDqymw9cfL7csAYS0cDYH/view?usp=sharing>

Now the datasets structure should look like

```
DSNet
└── datasets/
    ├── eccv16_dataset_ovp_google_pool5.h5
    ├── eccv16_dataset_summe_google_pool5.h5
    ├── eccv16_dataset_tvsum_google_pool5.h5
    ├── eccv16_dataset_youtube_google_pool5.h5
    └── readme.txt
```

## Pre-trained Models

Our pre-trained models are now available online. You may download them for evaluation, or you may skip this section and train a new one from scratch.

```sh
mkdir -p models && cd models
# anchor-based model
wget https://www.dropbox.com/s/0jwn4c1ccjjysrz/pretrain_ab_basic.zip
unzip pretrain_ab_basic.zip
# anchor-free model
wget https://www.dropbox.com/s/2hjngmb0f97nxj0/pretrain_af_basic.zip
unzip pretrain_af_basic.zip
```

To evaluate our pre-trained models, type:

```sh
# evaluate anchor-based model
python evaluate.py anchor-based --model-dir ../models/pretrain_ab_basic/ --splits ../splits/tvsum.yml ../splits/summe.yml
# evaluate anchor-free model
python evaluate.py anchor-free --model-dir ../models/pretrain_af_basic/ --splits ../splits/tvsum.yml ../splits/summe.yml --nms-thresh 0.4
```

If everything works fine, you will get similar F-score results as follows.

|              | TVSum | SumMe |
| ------------ | ----- | ----- |
| Anchor-based | 62.05 | 50.19 |
| Anchor-free  | 61.86 | 51.18 |

## Training

### Anchor-based

To train anchor-based attention model on TVSum and SumMe datasets with canonical settings, run

```sh
python train.py anchor-based --model-dir ../models/ab_basic --splits ../splits/tvsum.yml ../splits/summe.yml
```

To train on augmented and transfer datasets, run

```sh
python train.py anchor-based --model-dir ../models/ab_tvsum_aug/ --splits ../splits/tvsum_aug.yml
python train.py anchor-based --model-dir ../models/ab_summe_aug/ --splits ../splits/summe_aug.yml
python train.py anchor-based --model-dir ../models/ab_tvsum_trans/ --splits ../splits/tvsum_trans.yml
python train.py anchor-based --model-dir ../models/ab_summe_trans/ --splits ../splits/summe_trans.yml
```

To train with LSTM, Bi-LSTM or GCN feature extractor, specify the `--base-model` argument as `lstm`, `bilstm`, or `gcn`. For example,

```sh
python train.py anchor-based --model-dir ../models/ab_basic --splits ../splits/tvsum.yml ../splits/summe.yml --base-model lstm
```

### Anchor-free

Much similar to anchor-based models, to train on canonical TVSum and SumMe, run

```sh
python train.py anchor-free --model-dir ../models/af_basic --splits ../splits/tvsum.yml ../splits/summe.yml --nms-thresh 0.4
```

Note that NMS threshold is set to 0.4 for anchor-free models.

## Evaluation

To evaluate your anchor-based models, run

```sh
python evaluate.py anchor-based --model-dir ../models/ab_basic/ --splits ../splits/tvsum.yml ../splits/summe.yml
```

For anchor-free models, remember to specify NMS threshold as 0.4.

```sh
python evaluate.py anchor-free --model-dir ../models/af_basic/ --splits ../splits/tvsum.yml ../splits/summe.yml --nms-thresh 0.4
```

## Generating Shots with KTS

Based on the public datasets provided by [DR-DSN](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce), we apply [KTS](https://github.com/pathak22/videoseg/tree/master/lib/kts) algorithm to generate video shots for OVP and YouTube datasets. Note that the pre-processed datasets already contain these video shots. To re-generate video shots, run

```sh
python make_shots.py --dataset ../datasets/eccv16_dataset_ovp_google_pool5.h5
python make_shots.py --dataset ../datasets/eccv16_dataset_youtube_google_pool5.h5
```

## Acknowledgments

We gratefully thank the below open-source repo, which greatly boost our research.

+ Thank [KTS](https://github.com/pathak22/videoseg/tree/master/lib/kts) for the effective shot generation algorithm.
+ Thank [DR-DSN](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce) for the pre-processed public datasets.
+ Thank [VASNet](https://github.com/ok1zjf/VASNet) for the training and evaluation pipeline.

## Citation

If you find our codes or paper helpful, please consider citing.

```
@article{zhu2020dsnet,
  title={DSNet: A Flexible Detect-to-Summarize Network for Video Summarization},
  author={Zhu, Wencheng and Lu, Jiwen and Li, Jiahao and Zhou, Jie},
  journal={IEEE Transactions on Image Processing},
  volume={30},
  pages={948--962},
  year={2020}
}
```

```
DSNet
├─ .git
│  ├─ COMMIT_EDITMSG
│  ├─ config
│  ├─ description
│  ├─ FETCH_HEAD
│  ├─ HEAD
│  ├─ hooks
│  │  ├─ applypatch-msg.sample
│  │  ├─ commit-msg.sample
│  │  ├─ fsmonitor-watchman.sample
│  │  ├─ post-update.sample
│  │  ├─ pre-applypatch.sample
│  │  ├─ pre-commit.sample
│  │  ├─ pre-merge-commit.sample
│  │  ├─ pre-push.sample
│  │  ├─ pre-rebase.sample
│  │  ├─ pre-receive.sample
│  │  ├─ prepare-commit-msg.sample
│  │  ├─ push-to-checkout.sample
│  │  └─ update.sample
│  ├─ index
│  ├─ info
│  │  └─ exclude
│  ├─ logs
│  │  ├─ HEAD
│  │  └─ refs
│  │     ├─ heads
│  │     │  └─ main
│  │     └─ remotes
│  │        └─ origin
│  │           ├─ HEAD
│  │           └─ main
│  ├─ objects
│  │  ├─ 1b
│  │  │  └─ e28d4aaf6ca0a65190192e98f8c397d0478a68
│  │  ├─ 1c
│  │  │  └─ e9778314f7b2dd07681dc5296364f77cbb8e07
│  │  ├─ 28
│  │  │  └─ 8b55b7862ed8704f8ca0a3a6f649f9db33c756
│  │  ├─ 2f
│  │  │  └─ 37ce14b62cf7bdc6bb89f48d2fb562a1fa578a
│  │  ├─ 39
│  │  │  └─ 31fc2b073728c42c0c025391ac82401d0331fc
│  │  ├─ 3f
│  │  │  └─ 2036880b9e1cea10c06df7dccdb47179e82fcc
│  │  ├─ 40
│  │  │  └─ 5c3411c3518885121cd07bd8a098819f518996
│  │  ├─ 5f
│  │  │  └─ ce8cb6450c4cada1476ddb0d0eea7f89987306
│  │  ├─ 63
│  │  │  └─ 94d18db85029b0847b5ddd26c7f00f1204f04f
│  │  ├─ 69
│  │  │  └─ 66fe842a4794db780fd333a0a89bc306e2fa5d
│  │  ├─ 6b
│  │  │  └─ bded04a20140500119e36bb6b112f41faff6b2
│  │  ├─ 8d
│  │  │  └─ 878d2f4fd149402022256d14938094266550a4
│  │  ├─ 91
│  │  │  └─ 6e0a5dba34d0e222405fce4054af7a8d53d8e5
│  │  ├─ 92
│  │  │  └─ 75be2c561bbc151d34787ac4f00ffcfc180a97
│  │  ├─ 9f
│  │  │  └─ fc4e69d62de5be90ca99df594ba74119dbaba9
│  │  ├─ a9
│  │  │  └─ a5765a8c24c0359a9d3f7f3667775d6eed4bdc
│  │  ├─ b4
│  │  │  ├─ 096780ae8bf791c884c79ff0c3e1a34c786343
│  │  │  └─ 4dfab2d621389ba71e7f34e7affe3593b562e9
│  │  ├─ c5
│  │  │  └─ 9e19a861be43e0a9284dfffaeb0fe25d17685b
│  │  ├─ e5
│  │  │  └─ 2ac1c7155e7d77d2e8da05de353a5de6ac464a
│  │  ├─ info
│  │  └─ pack
│  │     ├─ pack-4d3d8fd211ad53c586189201257b46271c026feb.idx
│  │     └─ pack-4d3d8fd211ad53c586189201257b46271c026feb.pack
│  ├─ ORIG_HEAD
│  ├─ packed-refs
│  └─ refs
│     ├─ heads
│     │  └─ main
│     ├─ remotes
│     │  └─ origin
│     │     ├─ HEAD
│     │     └─ main
│     └─ tags
├─ .github
│  └─ workflows
│     └─ unit-test.yml
├─ .gitignore
├─ docs
│  └─ framework.jpg
├─ LICENSE
├─ README.md
├─ requirements.txt
├─ splits
│  ├─ readme.md
│  ├─ summe.yml
│  ├─ summe_aug.yml
│  ├─ summe_trans.yml
│  ├─ tvsum.yml
│  ├─ tvsum_aug.yml
│  └─ tvsum_trans.yml
├─ src
│  ├─ anchor_based
│  │  ├─ anchor_helper.py
│  │  ├─ dsnet.py
│  │  ├─ losses.py
│  │  └─ train.py
│  ├─ anchor_free
│  │  ├─ anchor_free_helper.py
│  │  ├─ dsnet_af.py
│  │  ├─ losses.py
│  │  └─ train.py
│  ├─ evaluate.py
│  ├─ helpers
│  │  ├─ bbox_helper.py
│  │  ├─ data_helper.py
│  │  ├─ init_helper.py
│  │  └─ vsumm_helper.py
│  ├─ kts
│  │  ├─ cpd_auto.py
│  │  ├─ cpd_nonlin.py
│  │  ├─ demo.py
│  │  ├─ LICENSE
│  │  └─ README.md
│  ├─ make_shots.py
│  ├─ make_split.py
│  ├─ modules
│  │  ├─ models.py
│  │  └─ model_zoo.py
│  ├─ output_file
│  │  └─ evaluation.txt
│  ├─ read_h5_file.py
│  └─ train.py
└─ tests
   ├─ anchor_based
   │  ├─ test_ab_losses.py
   │  ├─ test_anchor_helper.py
   │  └─ __init__.py
   ├─ anchor_free
   │  ├─ test_af_losses.py
   │  ├─ test_anchor_free_helper.py
   │  └─ __init__.py
   ├─ helpers
   │  ├─ test_bbox_helper.py
   │  ├─ test_data_helper.py
   │  ├─ test_vsumm_helper.py
   │  └─ __init__.py
   ├─ mock_run.sh
   ├─ modules
   │  ├─ test_models.py
   │  └─ __init__.py
   ├─ test_train.py
   └─ __init__.py

```