import os
import torch
import cv2
import json

from torch.utils import data
from tools.data_reading import CompressedWindowDataset

image_width = 128
image_height = 96
dataset = CompressedWindowDataset()

for i in range(len(dataset)):
    img, spectr, lab = dataset[i]
    