import random
from os import PathLike
from pathlib import Path
from typing import Any, List, Dict

import h5py
import numpy as np
import yaml

import cv2
import imageio

class VideoDataset(object):
    def __init__(self, keys: List[str]):
        self.keys = keys
        self.datasets = self.get_datasets(keys)

    def __getitem__(self, index):
        key = self.keys[index]
        video_path = Path(key)
        dataset_name = str(video_path.parent)
        video_name = video_path.name
        # video_file is a HDF5 group object
        video_file = self.datasets[dataset_name][video_name]
        ##print(video_file)

        # Load all data
        filename = video_file['filename'][...][()].decode("utf-8")
        n_frames = video_file['n_frames'][...].astype(np.int32)
        picks = video_file['picks'][...].astype(np.int32)
        gtscore = video_file['gtscore'][...].astype(np.float32)
        gtscore -= gtscore.min()
        gtscore /= gtscore.max()
        user_summary = None
        if 'user_summary' in video_file:
            user_summary = video_file['user_summary'][...].astype(np.float32)
        gtsummary = video_file['gtsummary'][...].astype(np.float32)

        seq_lenet = video_file['seq_lenet'][...].astype(np.float32)
        seq_alexnet = video_file['seq_alexnet'][...].astype(np.float32)
        seq_mobilenet = video_file['seq_mobilenet'][...].astype(np.float32)
        seq_squeeze = video_file['seq_squeeze'][...].astype(np.float32)
        seq_resnet = video_file['seq_resnet'][...].astype(np.float32)

        seq_lenet_c = video_file['seq_lenet_c'][...].astype(np.float32)
        seq_alexnet_c = video_file['seq_alexnet_c'][...].astype(np.float32)
        seq_mobilenet_c = video_file['seq_mobilenet_c'][...].astype(np.float32)
        seq_squeeze_c = video_file['seq_squeeze_c'][...].astype(np.float32)
        seq_resnet_c = video_file['seq_resnet_c'][...].astype(np.float32)

        cps_osg = video_file['change_points_osg'][...].astype(np.int32)
        cps_osg_sem = video_file['change_points_osg_sem'][...].astype(np.int32)
        cps_pyths = video_file['change_points_pyths'][...].astype(np.int32)
        cps_lenet = video_file['cps_lenet'][...].astype(np.int32)
        cps_alexnet = video_file['cps_alexnet'][...].astype(np.int32)
        cps_mobilenet = video_file['cps_mobilenet'][...].astype(np.int32)
        cps_squeeze = video_file['cps_squeeze'][...].astype(np.int32)
        cps_resnet = video_file['cps_resnet'][...].astype(np.int32)

        cps_lenet_c = video_file['cps_lenet_c'][...].astype(np.int32)
        cps_alexnet_c = video_file['cps_alexnet_c'][...].astype(np.int32)
        cps_mobilenet_c = video_file['cps_mobilenet_c'][...].astype(np.int32)
        cps_squeeze_c = video_file['cps_squeeze_c'][...].astype(np.int32)
        cps_resnet_c = video_file['cps_resnet_c'][...].astype(np.int32)

        cps_default = video_file['change_points'][...].astype(np.int32)
        seq_default = video_file['features'][...].astype(np.float32)
        nfps_default = video_file['n_frame_per_seg'][...].astype(np.int32)
        
        return filename, key, n_frames, picks, gtscore, user_summary, gtsummary, \
                seq_default, cps_default, nfps_default, \
                seq_lenet, seq_alexnet, seq_mobilenet, seq_squeeze, seq_resnet, \
                seq_lenet_c, seq_alexnet_c, seq_mobilenet_c, seq_squeeze_c, seq_resnet_c, \
                cps_lenet_c, cps_alexnet_c, cps_mobilenet_c, cps_squeeze_c, cps_resnet_c, \
                cps_osg, cps_osg_sem, cps_pyths, cps_lenet, cps_alexnet, cps_mobilenet, cps_squeeze, cps_resnet

    def __len__(self):
        return len(self.keys)

    @staticmethod
    def get_datasets(keys: List[str]) -> Dict[str, h5py.File]:
        dataset_paths = {str(Path(key).parent) for key in keys}
        ##print(keys)
        datasets = {path: h5py.File(path, 'r') for path in dataset_paths}
        return datasets


class DataLoader(object):
    def __init__(self, dataset: VideoDataset, shuffle: bool):
        self.dataset = dataset
        self.shuffle = shuffle
        self.data_idx = list(range(len(self.dataset)))

    def __iter__(self):
        self.iter_idx = 0
        if self.shuffle:
            random.shuffle(self.data_idx)
        return self

    def __next__(self):
        if self.iter_idx == len(self.dataset):
            raise StopIteration
        curr_idx = self.data_idx[self.iter_idx]
        batch = self.dataset[curr_idx]
        self.iter_idx += 1
        return batch


class AverageMeter(object):
    def __init__(self, *keys: str):
        self.totals = {key: 0.0 for key in keys}
        self.counts = {key: 0 for key in keys}

    def update(self, **kwargs: float) -> None:
        for key, value in kwargs.items():
            self._check_attr(key)
            self.totals[key] += value
            self.counts[key] += 1

    def __getattr__(self, attr: str) -> float:
        self._check_attr(attr)
        total = self.totals[attr]
        count = self.counts[attr]
        return total / count if count else 0.0

    def _check_attr(self, attr: str) -> None:
        assert attr in self.totals and attr in self.counts


def get_ckpt_dir(model_dir: PathLike) -> Path:
    return Path(model_dir) / 'checkpoint'


def get_ckpt_path(model_dir: PathLike,
                  split_path: PathLike,
                  split_index: int) -> Path:
    split_path = Path(split_path)
    return get_ckpt_dir(model_dir) / f'{split_path.name}.{split_index}.pt'


def load_yaml(path: PathLike) -> Any:
    with open(path) as f:
        obj = yaml.safe_load(f)
    return obj


def dump_yaml(obj: Any, path: PathLike) -> None:
    with open(path, 'w') as f:
        yaml.dump(obj, f)

def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 

def open_video(video_name, video_path, sampling_interval, printing = True):
	if printing:
		print("\nOpening " + str(video_name[:-4]) + " video! \n")
	
	video = imageio.get_reader(video_path + "\\" + video_name)
	n_frame_video = video.count_frames()
	
	# Read all the frames
	idx_frames = list(range(0,n_frame_video-1,1))
	frames = [video.get_data(i) for i in range(0,n_frame_video)]

    # Frames subset
	idx_frames_sel = idx_frames[::sampling_interval]
	frames_sel = frames[::sampling_interval]

	if printing:
		print ("\tLength of video %d" % n_frame_video)
		print ("\tConsidered frames %d" % len(frames_sel))
		print ("\n")
	return video, frames, frames_sel, n_frame_video

# Function for writing the video from a sequence of frames
def write_video_from_frame(output_path, video_name, model_name, summ_frames, printing=True):
	if printing:
		print("\n Writing the video summary..")
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	height, width, channels = summ_frames[0].shape
	filename = output_path + "\\" + video_name[:-4] + "_" + model_name + ".mp4"
	out = cv2.VideoWriter(filename, fourcc, 20, (width, height))
	for frame in summ_frames:
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		out.write(frame)
	out.release()
	if printing:
		print("\t Video summary of " + str(video_name[:-4]) + " saved \n")
		print("\t " + filename)
	return None
