r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

"""
import sys
sys.path.append('../')
from datetime import datetime
import os
import time

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


def main():
    checkpoint_root = os.path.join('..','checkpoints',datetime.today().strftime('%Y-%m-%d'))
    if not os.path.isdir(checkpoint_root):
        os.mkdir(checkpoint_root)

    # Use  CUDA_AVAILABLE_DEVICES=0 to control which device to use
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(f'using device {device}')
    # Data loading code
    print("Loading data")

    dataset, num_classes = get_dataset("windowpolar", "train", get_transform(train=True), os.path.join("..","data"))
    dataset_test, _ = get_dataset("windowpolar", "val", get_transform(train=False), os.path.join("..","data"))

    img, spec, lab = dataset[0]
    print(img.shape)
    print(spec.shape)
    print(lab)

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
    print(images[0].shape)
    print(images[1].shape)

    print("Creating model")
    model = build_rpn_injection_model()
    model.to(device)

    model_without_ddp = model

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=0.001/8, momentum=0.9, weight_decay=1e-4)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 90], gamma=0.1)

    resume = ''

    if resume:
        checkpoint = torch.load(resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    print("Start training")
    start_time = time.time()

    #evaluate(model, data_loader_test, device=device)
    utils.save_on_master({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            },
            os.path.join(checkpoint_root, 'model_rpninjection_{}.pth'.format(-1)))
    epochs = 100
    train_print_freq = 10

    for epoch in range(epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, train_print_freq)
        lr_scheduler.step()
        utils.save_on_master({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            },
            os.path.join(checkpoint_root, 'model_rpninjection_{}.pth'.format(epoch)))

        # evaluate after every epoch
        evaluate(model, data_loader_test, device=device)


    total_time = time.time() - start_time
    print('Training time {} seconds'.format(total_time))


if __name__ == "__main__":
    main()
