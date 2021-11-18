r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

"""
import sys
sys.path.append('../')
import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn

from tools.coco_utils import get_coco, get_coco_kp
from tools.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from tools.engine import train_one_epoch, coco_evaluate

from tools import utils
import tools.transforms as T


def get_dataset(name, image_set, transform, data_path):
    p, ds_fn, num_classes = data_path, get_coco, 91
    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    # Use  CUDA_AVAILABLE_DEVICES=0 to control which device to use
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(f'using device {device}')
    # Data loading code
    print("Loading data")

    dataset, num_classes = get_dataset("coco", "largetrain", get_transform(train=True), "coco")
    dataset_test, _ = get_dataset("coco", "minival", get_transform(train=False), "coco")

    print("Creating data loaders")
    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    batch_size = 2

    train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=8,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=8,
        collate_fn=utils.collate_fn)

    print("Creating model")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=num_classes,
                                                              pretrained=False)
    model.to(device)

    model_without_ddp = model

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=0.02/8, momentum=0.9, weight_decay=1e-4)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[16, 22], gamma=0.1)

    resume = ''

    if resume:
        checkpoint = torch.load(resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    print("Start training")
    start_time = time.time()

    #coco_evaluate(model, data_loader_test, device=device)
    utils.save_on_master({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            },
            os.path.join('checkpoints', 'model_baseclass_{}.pth'.format(1)))
    epochs = 26
    train_print_freq = 1000

    for epoch in range(epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, train_print_freq)
        lr_scheduler.step()
        utils.save_on_master({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            },
            os.path.join('checkpoints', 'model_baseclass_{}.pth'.format(epoch)))

        # evaluate after every epoch
        coco_evaluate(model, data_loader_test, device=device)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    main()
