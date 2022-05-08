import sys
sys.path.append('../')
from toolkits.imgproc import vid_read
import os
from scipy import signal
import cv2
import json
import numpy as np

'''
converts YOLO annotations of each 10th frame in the video to COCO annotations and generate spectrograms.
'''
anno_root = os.path.join('..','data','annotations')
img_root = os.path.join('..','data','images')
lab_root = os.path.join('..','data','labels')
vid_root = os.path.join('..','data','videos')
spec_root = os.path.join('..','data','spectrograms')
img_height = 128
img_width = 72
fps = 30.0

def pool_img(image, new_width, new_height):
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

anno = {'images':[], 'spectrograms':[], 'categories': [{"supercategory": "window","id": 1,"name": "window"}], 'annotations':[]}
img_id = 1
spec_id = 1
box_id = 1
for lab_folder in os.listdir(lab_root):
	# each folder corresponds to a video
	# lab_folder should correspond to generic name of video
	if not any(char.isdigit() for char in lab_folder): continue
	vid_file = os.path.join(vid_root,lab_folder+'.mov') # assume .mov type
	print(vid_file)
	assert(os.path.exists(vid_file))
	vid = vid_read(vid_file)
	vid = [pool_img(frame, img_width, img_height) for frame in vid]
	vid = np.stack(vid, axis=0)
	freqs, times, spectr = signal.spectrogram(vid, fs=fps, window=('hamming'), noverlap=12, nperseg=14, axis=0, mode='magnitude') 
	# all_frame has shape (T,16,9,3) (T)--how many frames are in the video
	# output (N, 16, 9, 3, T') -- applies fourier analysis to each all_shape[start:end,i,j,k] --> (N)
	# output (N, 16, 9, 3, T')
	print(spectr.shape)
	assert(len(spectr) == 8) 
	spectr = np.transpose(spectr, (4,1,2,3,0)) # (T',16,9,3,8)
	spectr = np.reshape(spectr, (spectr.shape[0], spectr.shape[1], spectr.shape[2], spectr.shape[3]*spectr.shape[4]))
	labs_files = os.listdir(os.path.join(lab_root, lab_folder))
	labs_files = [file for file in labs_files if '.txt' in file]
	labs_files.sort()
	vid_len = len(vid)
	for file in labs_files:
		i = int(file.split('.')[0])*10
		speci = 0
		timed = 1e9
		ftime = i/fps
		for i, t in enumerate(times):
			if abs(t-ftime)<timed:
				timed = abs(t-ftime)
				speci = i
		# print(times)
		# print(times[speci], ftime)
		img_file_path = os.path.join(lab_folder,os.path.splitext(file)[0]+'.jpg')
		img = cv2.imread(os.path.join(img_root, img_file_path))
		width = img.shape[1]
		height = img.shape[0]
		anno['images'].append({'file_name':img_file_path, 'height':height, 'width':width, 'id':img_id})
		del img

		cur_spectr = spectr[speci]
		if not os.path.isdir(os.path.join(spec_root, lab_folder)):
			os.mkdir(os.path.join(spec_root, lab_folder))
		spec_file_path = os.path.join(lab_folder, os.path.splitext(file)[0]+'.npy')
		np.save(os.path.join(spec_root, spec_file_path), cur_spectr)
		del cur_spectr
		anno['spectrograms'].append({'file_name':spec_file_path, 'id':spec_id})

		with open(os.path.join(lab_root,lab_folder,file),'r') as f:
			oboxes = f.readlines()
		boxes = []
		for box in oboxes:
			box = [float(i) for i in box.split()]
			box[1] *= width
			box[3] *= width
			box[2] *= height
			box[4] *= height
			boxes.append([box[1]-box[3]/2, box[2]-box[4]/2, box[3], box[4]])

		for box in boxes:
			anno['annotations'].append({'iscrowd':0, 'image_id':img_id, 'spectrogram_id':spec_id, 'category_id':1, 'id':box_id, 'bbox':box, 'area':box[2]*box[3]})
			box_id+=1

		img_id+=1
		spec_id+=1
	del spectr

with open(os.path.join(anno_root, 'annotations_27.json'), 'w') as f:
	json.dump(anno, f)

