import sys
sys.path.append('../')
from datetime import datetime
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data
from torch import nn

from toolkits.coco_utils import get_coco
from toolkits.engine import train_one_epoch, evaluate
from model_zoo.RPNInjection import RPNInjection, build_rpn_injection_model
from toolkits import utils
import toolkits.transforms as T

def get_dataset(name, image_set, transform, data_path):
    p, ds_fn, num_classes = data_path, get_coco, 2
    ds = ds_fn(p, mode=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

print("Loading data")

dataset, num_classes = get_dataset("windowpolar", "train", get_transform(train=True), os.path.join("..","data"))
dataset_test, _ = get_dataset("windowpolar", "val", get_transform(train=False), os.path.join("..","data"))

img, spec, lab = dataset[0]
# exit()

print("Creating data loaders")
train_sampler = torch.utils.data.RandomSampler(dataset)
test_sampler = torch.utils.data.SequentialSampler(dataset_test)

batch_size = 2

train_batch_sampler = torch.utils.data.BatchSampler(
		train_sampler, batch_size, drop_last=True)

data_loader = torch.utils.data.DataLoader(
	dataset, batch_sampler=train_batch_sampler, num_workers=6,
	collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
	dataset_test, batch_size=1,
	sampler=test_sampler, num_workers=6,
	collate_fn=utils.collate_fn)

images, spectrs, targets = next(iter(data_loader))
images = np.transpose(images[0].numpy(), (1, 2, 0))
spectrs = np.transpose(spectrs[0].numpy(), (1, 2, 0))

fig, ax = plt.subplots(1, 2)

ax[0].pcolormesh(list(range(spectrs.shape[1])), list(range(spectrs.shape[0])), spectrs[:,:,9], shading='nearest')
ax[0].invert_yaxis()
ax[1].imshow(images)
print(spectrs.shape)

plt.show()

