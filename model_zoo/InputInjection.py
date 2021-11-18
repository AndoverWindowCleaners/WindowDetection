import sys
sys.path.append('../')
import torch
from torch import nn, Tensor
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from model_zoo.PolarizePreprocess import PolarizePreprocess
import torch.nn.functional as F
import warnings
from collections import OrderedDict
from typing import Tuple, List, Dict, Optional, Union

class PolarizeBackbone(nn.Module):
	def __init__(self, backbone):
		super(PolarizeBackbone, self).__init__()
		self.backbone = backbone
		newConv1 = nn.Conv2d(5, 64, (7,7), stride=(2,2), padding=(3,3), bias=False)
		newConv1.weight.data[:, :3, :, :] = self.backbone.body.conv1.weight.data
		self.backbone.body.conv1 = newConv1

	def __call__(self, x):
		'''
		x is expected to be (batch, channel, h, w)
		'''
		return self.backbone(x)		

class InputInjection(nn.Module):
	def __init__(self, pretrained=True):
		super(InputInjection, self).__init__()
		def fasterRCNNforward(self, images, polars, targets=None):
			if self.training and targets is None:
				raise ValueError("In training mode, targets should be passed")
			if self.training:
				assert targets is not None
				for target in targets:
					boxes = target["boxes"]
					if isinstance(boxes, torch.Tensor):
						if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
							raise ValueError(
								"Expected target boxes to be a tensor" "of shape [N, 4], got {:}.".format(boxes.shape)
							)
					else:
						raise ValueError("Expected target boxes to be of type " "Tensor, got {:}.".format(type(boxes)))

			original_image_sizes: List[Tuple[int, int]] = []
			for img in images:
				val = img.shape[-2:]
				assert len(val) == 2
				original_image_sizes.append((val[0], val[1]))

			images, targets = self.transform(images, targets)
			# Check for degenerate boxes
			if targets is not None:
				for target_idx, target in enumerate(targets):
					boxes = target["boxes"]
					degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
					if degenerate_boxes.any():
						# print the first degenerate box
						bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
						degen_bb: List[float] = boxes[bb_idx].tolist()
						raise ValueError(
							"All bounding boxes should have positive height and width."
							" Found invalid box {} for target at index {}.".format(degen_bb, target_idx)
						)
			imgs = images.tensors
			if polars.shape[2] != imgs.shape[2] or polars.shape[3] != imgs.shape[3]:
				polars = F.interpolate(polars, (imgs.shape[2],imgs.shape[3]), mode='bilinear')
			inputs = torch.cat([imgs, polars], dim=1) # they are not the same dims
			features = self.backbone(inputs)
			if isinstance(features, torch.Tensor):
				features = OrderedDict([("0", features)])
			proposals, proposal_losses = self.rpn(images, features, targets)
			detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
			detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

			losses = {}
			losses.update(detector_losses)
			losses.update(proposal_losses)

			if torch.jit.is_scripting():
				if not self._has_warned:
					warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
					self._has_warned = True
				return losses, detections
			else:
				return self.eager_outputs(losses, detections)

		self.polarPrep = PolarizePreprocess()
		FasterRCNN.forward = fasterRCNNforward
		self.fasterRCNN = torchvision.models.detection.fasterrcnn_resnet50_fpn(\
			pretrained=pretrained, pretrained_backbone=True, \
			image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], \
		)
		in_features = self.fasterRCNN.roi_heads.box_predictor.cls_score.in_features
		self.fasterRCNN.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
		self.fasterRCNN.backbone = PolarizeBackbone(self.fasterRCNN.backbone)
		for param in self.fasterRCNN.parameters():
		    param.requires_grad = True
		for param in self.fasterRCNN.backbone.parameters():
		    param.requires_grad = False

	def save(self, file_name="InputInjection.weights"):
		torch.save(self.state_dict(), file_name)
	
	def load(self, file_name="InputInjection.weights"):
		self.load_state_dict(torch.load(file_name))

	def forward(self,imgs,polars,labels=None):
		'''
		both imgs and polars ought to be tensors with
		(batch, depth, height, width)
		'''
		polars = self.polarPrep(polars)
		#print(polars.shape, imgs.shape)
		return self.fasterRCNN(imgs, polars, labels)

# model = InputInjection()
# model.eval()
# print(model)
# x = torch.ones((1,3,96,128))
# p = torch.ones((1,24,9,12))
# y = model(x,p)
# # print(y[0])

# given the pretrained nature of the model, I suspect zero initialization will be the best
