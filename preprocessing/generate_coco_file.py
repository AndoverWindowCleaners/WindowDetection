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
	# all_frame has shape (T,16,9,3) (T)--how many frames are in the video
	# output (N, 16, 9, 3, T') -- applies fourier analysis to each all_shape[start:end,i,j,k] --> (N)
	# output (N, 16, 9, 3, T')
	assert(len(spectr) == 8) 
	spectr = np.transpose(spectr, (4,1,2,3,0)) # (T',16,9,3,8)
	spectr = np.reshape(spectr, (spectr.shape[0], spectr.shape[1], spectr.shape[2], spectr.shape[3]*spectr.shape[4]))
	labs_files = os.listdir(os.path.join(lab_root, lab_folder))
	labs_files = [file for file in labs_files if '.txt' in file]
	labs_files.sort()
	for i,file in enumerate(labs_files):
		perc = i/len(labs_files)
		vidi = round(perc * len(vid))
		speci = round(perc * len(spectr))
		with open(os.path.join(lab_root,lab_folder,file),'r') as f:
			oboxes = f.readlines()
		boxes = []
		for box in oboxes:
			boxes.append([box[1]-box[3]/2, box[2]-box[4]/2, box[1]+box[3]/2, box[2]+box[4]/2])
		

