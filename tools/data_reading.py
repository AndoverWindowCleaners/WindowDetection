import numpy as np
import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.datasets.coco import CocoDetection
import pickle

class WindowDataset(CocoDetection):
    def __init__(self, data_path = None, labels_folder = None, images_folder = None):
        super(Dataset, self).__init__()
        if data_path is not None:
            with open(data_path,'rb') as f:
                self.data = pickle.load(f)
            return
        self.data = []
        image_width = 128
        image_height = 96
        for folder in os.listdir(labels_folder):
            if folder != '.DS_Store':
                for filename in os.listdir(f"{labels_folder}{folder}"):
                    if filename.endswith("YOLO"):
                        for _file in os.listdir(f"{labels_folder}{folder}/{filename}"):
                            self.data.append([None,None])
                            with open(f"{labels_folder}{folder}/{filename}/{_file}") as f:
                                boxes = []
                                labels = []
                                for line in f:
                                    line = line.split()
                                    label, rest = int(line[0]), line[1:]
                                    xmin, ymin, w, h = map(float, rest)
                                    xmin, ymin = int(xmin*image_width), int(ymin*image_height)
                                    w, h = int(w*image_width), int(h*image_height)
                                    boxes.append(torch.tensor([xmin, ymin, min(w+xmin, image_width-1), min(h+ymin, image_height-1)]))
                                    labels.append(label)
                                self.data[-1][1] = {'boxes':torch.stack(boxes).float(), 'labels':torch.tensor(labels).long()+1}
                            image = torch.tensor(cv2.cvtColor(cv2.imread(f"{images_folder}/{folder}/{_file[:-3]}jpg", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
                            image = image.permute(2,0,1).double()
                            self.data[-1][0] = image
        
    def __getitem__(self, idx):
        return tuple(self.data[idx])
    
    def __len__(self):
        return len(self.data)
