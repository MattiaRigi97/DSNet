
from torch import  nn
import numpy as np
import cv2

"""
# Define a feature extractor
class FeatureExtractor(nn.Module):
  def __init__(self, model):
    super(FeatureExtractor, self).__init__()
    # Extract LeNet AdaptiveAvgPool2d 
    self.pooling = nn.Sequential(*list(model.children())[:-2])
  
  def forward(self, x):
	# It will take the input 'x' until it returns the feature vector called 'out'
    out = self.pooling(x)
    return out """

# Function for generating the BGR hist of the selected frames
def generate_bgr_hist(frames, num_bins):
    print ("\t Generating linear Histrograms using OpenCV")
    channels=['b','g','r']
    hist=[]
    for frame in frames:
        feature_value = [cv2.calcHist([frame],[i],None,[num_bins],[0,256]) for i,col in enumerate(channels)]
        feature_value = np.asarray(feature_value).flatten()
        hist.append(feature_value)
    hist = np.asarray(hist)
    hist = (hist - hist.min()) / (hist.max() - hist.min())
    print ("\t Done generating!")
    print ("\t Shape of histogram: " + str(hist.shape))
    print ("\n")
    return hist

#from matplotlib import cm
#from PIL import Image
#import imageio
#from torchvision import models, transforms
#from helpers import init_helper, data_helper, vsumm_helper, bbox_helper
#from helpers.data_helper import open_video
#from modules.model_zoo import get_model
#from torch import optim, nn

# Initialize the model
# model = models.googlenet(pretrained=True)
# print(model(torch.randn(1,3,224,224)).shape)
# print(model)

# Initialize the feature extractor model
# new_model = FeatureExtractor(model)
# print(new_model(torch.randn(1,3,224,224)).shape)
# print(new_model)

# Change the device to GPU
# device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
# new_model = new_model.to(device)

# filename = "C:/Users/matti/Pictures/pannello_gabri.jpg"
# input_image = Image.open(filename)
# print(input_image.shape)
# print(input_image)

# sample execution (requires torchvision)
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# input_tensor = preprocess(input_image)
# input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available

# if torch.cuda.is_available():
#     input_batch = input_batch.to('cuda')
#     model.to('cuda')

# with torch.no_grad():
#     output = new_model(input_batch)

# print(output[0].shape)

# Open Video
# video_path = r"C:\Users\matti\OneDrive - Universita degli Studi di Milano-Bicocca\Uni\LM-DataScience\Tesi+Stage\Dataset\TVSum\ydata-tvsum50-v1_1\ydata-tvsum50-v1_1\video"
# video_name = "_xMr-HKMfVA.mp4"
# video, frames, idx_frames, n_frame_video = open_video(video_name, video_path, sampling_interval=15)
# print(frames[0])
# print(frames[0].shape)

# seq = np.asarray([])
# for frame in frames:
#     im = Image.fromarray(frame)
#     input_tensor = preprocess(im)
#     input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

#     # move the input and model to GPU for speed if available
#     if torch.cuda.is_available():
#         input_batch = input_batch.to('cuda')
#         model.to('cuda')

#     with torch.no_grad():
#         output = new_model(input_batch)
#     #print(output[0].cpu().shape)
#     seq = np.append(seq, output[0].cpu())

# print(seq.shape)
# b = np.reshape(seq, (-1,1024))
# print(b.shape)

