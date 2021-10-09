import numpy as np
import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.datasets.coco import CocoDetection
import pickle
from scipy import signal
from bisect import bisect_left

class WindowDataset(CocoDetection):
    def read_video(file_name):
        cap = cv2.VideoCapture(file_name)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        buf = np.empty((frameCount, 3, frameHeight, frameWidth), np.dtype('uint8'))
        fc = 0
        ret = True
        SAMPLING_FREQUENCY = cap.get(cv2.CAP_PROP_FPS)
        print(frameCount/SAMPLING_FREQUENCY)
        while (fc < frameCount  and ret):
            ret, img = cap.read()
            buf[fc] = np.transpose(img, (2,0,1))
            fc += 1
        cap.release()
        freqs, times, spectr = signal.spectrogram(buf, fs=SAMPLING_FREQUENCY, window=('hamming'), noverlap=13, nperseg=14, axis=0, mode='magnitude')
        return freqs, times, spectr, frameCount, SAMPLING_FREQUENCY

    def __init__(self, video_folder = None, labels_folder = None, images_folder = None):
        super(Dataset, self).__init__()
        self.video_folder = video_folder
        self.labels_folder = labels_folder
        self.images_folder = images_folder
        self.videos = [folder for folder in os.listdir(video_folder) if 'NW' in folder]
        self.image_width = 128
        self.image_height = 96
        
    def __getitem__(self, idx):
        video_name = self.videos[idx]
        name = video_name.split('.')[0]
        freqs, times, spectr, total_frame, fps = self.read_video(name)
        # spectr (F, H, W, C, T)
        dure = total_frame/fps
        folder = 'Folder '+name
        labels = None
        image = None
        all_spectr = []
        if 'NW' not in folder:
            for filename in os.listdir(f"{self.labels_folder}{folder}"):
                if filename.endswith("YOLO"):
                    _file_names = os.listdir(f"{self.labels_folder}{folder}/{filename}")
                    _file_names.sort()
                    for i, _file in enumerate(_file_names):
                        self.data.append([None,None])
                        with open(f"{self.labels_folder}{folder}/{filename}/{_file}") as f:
                            boxes = []
                            labels = []
                            for line in f:
                                line = line.split()
                                label, rest = int(line[0]), line[1:]
                                xcenter, ycenter, w, h = map(float, rest)
                                xcenter, ycenter = int(xcenter*self.image_width), int(ycenter*self.image_height)
                                xmin, ymin = xcenter-int(w/2*self.image_width), ycenter-int(h/2*self.image_height)
                                boxes.append(torch.tensor([xmin, ymin, w, h]))
                                labels.append(label)
                            labels = {'boxes':torch.stack(boxes).float(), 'labels':torch.tensor(labels).long()+1}
                        front_part = i*10/total_frame*dure
                        pos = bisect_left(times,front_part,0,len(times))
                        left = 0
                        if pos > 0:
                            left = times[pos-1]
                        right = times[pos]
                        assert(left<=front_part and front_part<=right)
                        leftW = (right-front_part)/(right-left) # this should weight left
                        rightW = (front_part-left)/(right-left) # this should weight right
                        cur_spectr = spectr[:,:,:,:,pos-1]*leftW + spectr[:,:,:,:,pos]*rightW
                        all_spectr.append(cur_spectr)
                        image = torch.tensor(cv2.cvtColor(cv2.imread(f"{self.images_folder}/{folder}/{_file[:-3]}jpg", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
                        image = image.permute(2,0,1).double()
        all_spectr = torch.stack(all_spectr,dim=0)
        # spectr (N, F, H, W, C)
        all_spectr = torch.flatten(all_spectr.permute(0,2,3,4,1), start_dim=3)
        # spectr (N, H, W, C*F)
        return image, all_spectr, labels
    
    def __len__(self):
        return len(self.videos)
