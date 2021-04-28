import argparse

import h5py
import numpy as np

from kts.cpd_auto import cpd_auto


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    h5in = h5py.File(args.dataset, 'r')
    h5out = h5py.File(args.dataset + '.custom', 'w')

    for video_name, video_file in h5in.items():
        features = video_file['features'][...].astype(np.float32)
        gtscore = video_file['gtscore'][...].astype(np.float32)
        gtsummary = video_file['gtsummary'][...].astype(np.float32)

        seq_len = gtscore.size
        n_frames = seq_len * 15 - 1 
        picks = np.arange(0, seq_len) * 15 # array([    0,    15,    30,    45,    60, ...])

        kernel = np.matmul(features, features.T) # Matrix product of two arrays
        change_points, _ = cpd_auto(kernel, seq_len - 1, 1) # Call of the KTS Function
        change_points *= 15
        change_points = np.hstack((0, change_points, n_frames))
        begin_frames = change_points[:-1]
        end_frames = change_points[1:]
        change_points = np.vstack((begin_frames, end_frames - 1)).T
        # Here, the change points are detected (Change-point positions t0, t1, ..., t_{m-1})

        # For each segment, calculate the number of frames
        n_frame_per_seg = end_frames - begin_frames
        
        # Add the video info on the new dataset (that contain shots)
        h5out.create_dataset(video_name + '/features', data=features)                   # add features
        h5out.create_dataset(video_name + '/gtscore', data=gtscore)                     # add ground truth score  
        # h5out.create_dataset(name + '/user_summary', data=data_of_name)   
        h5out.create_dataset(video_name + '/change_points', data=change_points)         # add the change points
        h5out.create_dataset(video_name + '/n_frame_per_seg', data=n_frame_per_seg)     # add number of frame for each segment
        h5out.create_dataset(video_name + '/n_frames', data=n_frames)                   # add number of video frames
        h5out.create_dataset(video_name + '/picks', data=picks)                         # add initial picks
        # h5out.create_dataset(video_name + '/n_steps', data=data_of_name)
        h5out.create_dataset(video_name + '/gtsummary', data=gtsummary)                 # add ground truth summary  
        # h5out.create_dataset(name + '/video_name', data=data_of_name)

    h5in.close()
    h5out.close()


if __name__ == '__main__':
    main()
