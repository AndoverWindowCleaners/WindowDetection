import torch
from torch import nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



class InputInjection(nn.Module):
	def __init__(self, pretrained=False):
		super(InputInjection, self).__init__()
		self.fasterRCNN = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained, pretrained_backbone=True)
		in_features = self.fasterRCNN.roi_heads.box_predictor.cls_score.in_features
		self.fasterRCNN.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
		for param in self.fasterRCNN.parameters():
		    param.requires_grad = True
		for param in self.fasterRCNN.backbone.parameters():
		    param.requires_grad = False
		print(self.fasterRCNN.backbone.body)

	def save(self, file_name="InputInjection.weights"):
		torch.save(self.fasterRCNN.state_dict(), file_name)

	def __call__(self,imgs,labels=None):
		return self.fasterRCNN.backbone.body(imgs)

model = InputInjection()
x = torch.zeros((1,3,265,265))
y = model(x)
print(y.keys())
print(y['0'].shape, y['1'].shape, y['2'].shape, y['3'].shape)

# create a separate FPN for the polarization info
# another approach is to attach the polarization info directly to the input data
# or something else we can do is to attach it to the ROI pooling layer 