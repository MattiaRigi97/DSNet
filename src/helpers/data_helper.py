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
        seq = video_file['features'][...].astype(np.float32)
        gtscore = video_file['gtscore'][...].astype(np.float32)

        cps_kts = video_file['change_points'][...].astype(np.int32)
        cps_osg = video_file['change_points_osg'][...].astype(np.int32)
        cps_osg_sem = video_file['change_points_osg_sem'][...].astype(np.int32)
        cps_pyths = video_file['change_points_pyths'][...].astype(np.int32)

        n_frames = video_file['n_frames'][...].astype(np.int32)
        nfps = video_file['n_frame_per_seg'][...].astype(np.int32)
        picks = video_file['picks'][...].astype(np.int32)
        user_summary = None
        if 'user_summary' in video_file:
            user_summary = video_file['user_summary'][...].astype(np.float32)

        gtscore -= gtscore.min()
        gtscore /= gtscore.max()

        return filename, key, seq, gtscore, cps_kts, cps_osg, cps_osg_sem, cps_pyths, \
        n_frames, nfps, picks, user_summary

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
