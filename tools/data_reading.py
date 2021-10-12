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
        buf = np.empty((frameCount, 3, 9, 12), np.dtype('uint8'))
        fc = 0
        ret = True
        SAMPLING_FREQUENCY = cap.get(cv2.CAP_PROP_FPS)
        print(frameCount/SAMPLING_FREQUENCY)
        while (fc < frameCount  and ret):
            ret, img = cap.read()
            buf[fc] = np.transpose(cv2.resize(img, (12,9)), (2,0,1))
            fc += 1
        cap.release()
        freqs, times, spectr = signal.spectrogram(buf, fs=SAMPLING_FREQUENCY, window=('hamming'), noverlap=13, nperseg=14, axis=0, mode='magnitude')
        return freqs, times, spectr, frameCount, SAMPLING_FREQUENCY

    def __init__(self, video_folder = 'data/training_videos/', labels_folder = 'data/labels/', images_folder = 'data/images/', spectr_folder = None):
        super(Dataset, self).__init__()
        self.video_folder = video_folder
        self.labels_folder = labels_folder
        self.images_folder = images_folder
        self.spectr_folder = spectr_folder
        self.videos = [vid for vid in os.listdir(video_folder) if 'W' in vid or 'NW' in vid]
        self.image_width = 128
        self.image_height = 96
        
    def gen_spectro(self, idx):
        video_name = self.videos[idx]
        name = video_name.split('.')[0]
        freqs, times, spectr, total_frame, fps = WindowDataset.read_video(f'{self.video_folder}{video_name}')
        # spectr (F, H, W, C, T)
        dure = total_frame/fps
        folder = 'Folder '+name
        all_spectr = []
        if os.path.exists(f"{self.labels_folder}{folder}") and os.path.exists(f"{self.images_folder}{folder}"):
            for filename in os.listdir(f"{self.labels_folder}{folder}"):
                if filename.endswith("YOLO"):
                    _file_names = os.listdir(f"{self.labels_folder}{folder}/{filename}")
                    _file_names.sort()
                    for i, _file in enumerate(_file_names):
                        if i == 0 or i == len(_file_names)-1:
                            continue
                        front_part = i*10/total_frame*dure
                        pos = bisect_left(times,front_part,0,len(times))
                        left = 0
                        if pos > 0:
                            left = times[pos-1]
                        right = times[-1]
                        if pos < len(times):
                            right = times[pos]
                        else:
                            print(times[-1], front_part)
                        assert(left<=front_part and front_part<=right)
                        leftW = (right-front_part)/(right-left) # this should weight left
                        rightW = (front_part-left)/(right-left) # this should weight right
                        cur_spectr = spectr[:,:,:,:,pos-1]*leftW + spectr[:,:,:,:,pos]*rightW
                        all_spectr.append(torch.tensor(cur_spectr))
        if len(all_spectr) == 0:
            return None, None, None
        all_spectr = torch.stack(all_spectr,dim=0)
        # spectr (N, F, H, W, C)
        all_spectr = torch.flatten(all_spectr.permute(0,2,3,4,1), start_dim=3)
        # spectr (N, H, W, C*F)
        return all_spectr

    def __getitem__(self, idx):
        video_name = self.videos[idx]
        name = video_name.split('.')[0]
        folder = 'Folder '+name
    
    def __len__(self):
        return len(self.videos)
    
    def save(self, spectr_folder = 'data/spectrograms/'):
        for idx in range(len(self.videos)):
            video_name = self.videos[idx]
            name = video_name.split('.')[0]
            if os.path.exists(f'{spectr_folder}/{name}.spectr'):
                continue
            imgs, spectrs, labs = self.gen_spectro(idx)
            if imgs is None or spectrs is None or labs is None:
                continue
            with open(f'{spectr_folder}/{name}.spectr', 'wb') as f:
                pickle.dump(spectrs, f)

class CompressedWindowDataset(CocoDetection):
    def __init__(self, video_folder = 'data/training_videos/', labels_folder = 'data/labels/', images_folder = 'data/images/', spectr_folder = None):
        super(Dataset, self).__init__()
        self.video_folder = video_folder
        self.labels_folder = labels_folder
        self.images_folder = images_folder
        self.spectr_folder = spectr_folder
        self.videos = [vid for vid in os.listdir(video_folder) if 'W' in vid or 'NW' in vid]
        self.image_width = 128
        self.image_height = 96