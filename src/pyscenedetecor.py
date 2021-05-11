import numpy as np
import math

import cv2
import imageio

import scenedetect
from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector

def set_scene_parameters(video_path, video_name):
    
    video = imageio.get_reader(video_path + "\\" + video_name)
    n_frame_video = video.count_frames()
    interval = math.trunc(n_frame_video/10)

    frames = [video.get_data(i * interval) for i in range(math.trunc(n_frame_video / interval))]
    
    channels = ['b','g','r']
    num_bins = 16
    hist=[]
    for frame in frames:
        feature_value = [cv2.calcHist([frame],[i],None,[num_bins],[0,256]) for i,col in enumerate(channels)]
        hist.append(np.asarray(feature_value).flatten())
    hist = np.asarray(hist)

    print(hist)
    print(hist.shape)

    frame_dim = frames[0].shape
    downscale_factor = round(frame_dim[1] / 224, 0)

    # Calcolare distanza tra frame e determinare soglia
    
    return threshold, downscale_factor

def find_scenes(video_path, threshold, downscale_factor):
    # Create our video & scene managers, then add the detector.
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    # Improve processing speed by downscaling before processing.
    video_manager.set_downscale_factor(downscale_factor)

    # Start the video manager and perform the scene detection.
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    # Each returned scene is a tuple of the (start, end) timecode.
    return scene_manager.get_scene_list()

