import torch
from torch import nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class PolarizeBackbone(nn.Module):
	def __init__(self, backbone):
		super(PolarizeBackbone, self).__init__()
		self.backbone = backbone
		newConv1 = nn.Conv2d(4, 64, (7,7), stride=(2,2), padding=(3,3), bias=False)
		newConv1.weight.data[:, :3, :, :] = self.backbone.body.conv1.weight.data
		self.backbone.body.conv1 = newConv1

	def __call__(self, x):
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

	def __call__(self,imgs,labels=None):
		return self.fasterRCNN(imgs)

model = InputInjection()
model.eval()
print(model)
x = torch.zeros((1,4,265,265))
y = model(x)
print(y[0])

# given the pretrained nature of the model, I suspect zero initialization will be the best
