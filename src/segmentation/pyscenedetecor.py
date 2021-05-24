
import scenedetect
from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.detectors import ThresholdDetector

def find_scenes(video_path, th, min_scene_length, min_perc):
    '''Function for split the video in shot'''
    # Create our video & scene managers, then add the detector.
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    
    # Improve processing speed by downscaling before processing.
    video_manager.set_downscale_factor() # Automatic determination of downscale factor (720,960) -> (180, 240) 
    frame_size = video_manager.get_framesize()
    downscale_factor = scenedetect.video_manager.compute_downscale_factor(frame_size[0]) # Compute the downscale factor utilized

    # Define the Detector
    # scene_manager.add_detector(ContentDetector(threshold=avg, min_scene_len = 300))
    scene_manager.add_detector(ThresholdDetector(threshold=th, min_percent = min_perc, min_scene_len = min_scene_length, 
                                                 fade_bias=0.0, add_final_scene=False, block_size=8))

    # Start the video manager and perform the scene detection.
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
        
    # Each returned scene is a tuple of the (start, end) timecode.
    return scene_manager.get_scene_list()


def mean_pixel_intensity_calc(video_path):
    '''Function for calculate mean pixel intensity for all frames in a video'''
    # Create our video & scene managers, then add the detector.
    video_manager = VideoManager([video_path])
    
    # Improve processing speed by downscaling before processing.
    video_manager.set_downscale_factor() # Automatic determination of downscale factor (720,960) -> (180, 240) 

    # Start the video manager and perform the scene detection.
    video_manager.start()
    
    # Read all the frames and compute the average pixel value/intensity for all pixels in a frame and return mean values     
    avg_list = []
    while True:
        ret_val, frame_image = video_manager.read()
        if ret_val:
            avg = int(scenedetect.detectors.threshold_detector.compute_frame_average(frame_image))
        if not ret_val:
            break
        avg_list.append(avg)
    
    all_mean_value = int(sum(avg_list)/len(avg_list))
        
    return all_mean_value