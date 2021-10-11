import torch
from torch import nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import cv2
import torch.nn.functional as F

class PolarizePreprocess(nn.Module):
    def __init__(self):
        # input is of shape (12,9,3*8)
        # we should upsample that to (128,96,2) or (128,96,1)
        super(PolarizePreprocess, self).__init__()
        self.conv1 = nn.Conv2d(24, 48, (3,3), padding='same') # gather info of neighbor cells
        self.conv2 = nn.Conv2d(48, 128, (1,1)) # blow up channels for classification
        self.conv4 = nn.Conv2d(128, 2, (1,1)) # final classifier
        # incorporate upconvs

    def __call__(self, x):
        '''
        x is expected to be (batch, channel, h, w)
        '''
        return self.backbone(x)		

class InputInjection(nn.Module):
    def __init__(self, pretrained=False):
        super(InputInjection, self).__init__()
        self.fasterRCNN = torchvision.models.detection.fasterrcnn_resnet50_fpn(\
            pretrained=pretrained, pretrained_backbone=True, \
            image_mean=[0.485, 0.456, 0.406, 0], image_std=[0.229, 0.224, 0.225, 1], \
        )
        in_features = self.fasterRCNN.roi_heads.box_predictor.cls_score.in_features
        self.fasterRCNN.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        self.fasterRCNN.backbone = PolarizeBackbone(self.fasterRCNN.backbone)
        for param in self.fasterRCNN.parameters():
            param.requires_grad = True
        for param in self.fasterRCNN.backbone.parameters():
            param.requires_grad = False

    def save(self, file_name="InputInjection.weights"):
        torch.save(self.fasterRCNN.state_dict(), file_name)

    def __call__(self,imgs,polars,labels=None):
        '''
        both imgs and polars ought to be tensors with
        (batch, depth, height, width)
        '''
        polars = F.interpolate(polars,(imgs.shape[-2], imgs.shape[-1])) # try different interpolations
        print(imgs.shape, polars.shape)
        x = torch.cat([imgs, polars], dim=1)
        return self.fasterRCNN(x)