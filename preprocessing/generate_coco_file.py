import sys
sys.path.append('../')
from toolkits.read_video import vid_read
import os
from scipy import signal
import cv2
import numpy as np

'''
converts YOLO annotations of each 10th frame in the video to COCO annotations and generate spectrograms.
'''
img_root = os.path.join('..','data','images')
lab_root = os.path.join('..','data','labels')
vid_root = os.path.join('..','data','videos')
spec_root = os.path.join('..','data','spectrograms')

def pool_img(image, new_width, new_height):
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

for lab_folder in os.listdir(lab_root):
	# each folder corresponds to a video
	# lab_folder should correspond to generic name of video
	if not any(char.isdigit() for char in lab_folder): continue
	vid_file = os.path.join(vid_root,lab_folder+'.mov') # assume .mov type
	assert(os.path.exists(vid_file))
	vid = vid_read(vid_file)
	vid = [pool_img(frame, 9, 16) for frame in vid]
	vid = np.stack(vid, axis=0)
	freqs, times, spectr = signal.spectrogram(vid, fs=30.0, window=('hamming'), noverlap=13, nperseg=14, axis=0, mode='magnitude') 
    # all_frame has shape (T,9,16,3) (T)--how many frames are in the video
    # output (N, 9, 16, 3, T') -- applies fourier analysis to each all_shape[start:end,i,j,k] --> (N)
    # output (N, 9, 16, 3, T')