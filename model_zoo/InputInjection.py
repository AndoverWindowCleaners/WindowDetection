import torch
from torch import nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from model_zoo.PolarizePreprocess import PolarizePreprocess
import torch.nn.functional as F

class PolarizeBackbone(nn.Module):
	def __init__(self, backbone):
		super(PolarizeBackbone, self).__init__()
		self.backbone = backbone
		newConv1 = nn.Conv2d(4, 64, (7,7), stride=(2,2), padding=(3,3), bias=False)
		newConv1.weight.data[:, :3, :, :] = self.backbone.body.conv1.weight.data
		self.backbone.body.conv1 = newConv1

	def __call__(self, x):
		'''
		x is expected to be (batch, channel, h, w)
		'''
		return self.backbone(x)		

class InputInjection(nn.Module):
	def __init__(self, pretrained=False):
		super(InputInjection, self).__init__()
		self.polarPrep = PolarizeBackbone()
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
		polars = self.polarPrep(polars)
		print(imgs.shape, polars.shape)
		x = torch.cat([imgs, polars], dim=1)
		return self.fasterRCNN(x)

model = InputInjection()
model.eval()
#print(model)
x = torch.zeros((1,3,265,265))
p = torch.zeros((1,1,265,265))
y = model(x,p)
print(y[0])

# given the pretrained nature of the model, I suspect zero initialization will be the best
