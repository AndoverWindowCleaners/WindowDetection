import sys
sys.path.append('../')
import torch
from torch import nn, Tensor
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torch.nn.functional as F
import warnings
from collections import OrderedDict
from typing import Tuple, List, Dict, Optional, Union


class PolarizePreprocess(nn.Module):
    def __init__(self):
        # video shape is 360 by 640 (W by H)
        # input is of shape (9,16,3*8) maybe another larger shape (18,32,3*8)
        # we should upsample that to (360,640,2)
        super(PolarizePreprocess, self).__init__()
        self.conv1 = nn.Conv2d(24, 72, (3,3), padding='same') # gather info of neighbor cells
        self.upconv1 = nn.ConvTranspose2d(72, 36, (4,3), stride=2, padding=0)
        self.upconv2 = nn.ConvTranspose2d(36, 36, (6,5), stride=3, padding=0) 
        self.upconv3 = nn.ConvTranspose2d(36, 36, (16,12), stride=6, padding=0) 
        # set up depthwise if needed
        self.conv2 = nn.Conv2d(36, 128, (1,1)) # blow up channels for classification
        self.conv3 = nn.Conv2d(128, 2, (1,1)) # final classifier
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        # maybe a good idea to use scale up module and back connections
    
    def save(self, file_name="PolarizePreprocess.weights"):
        torch.save(self.state_dict(), file_name)

    def forward(self, x):
        '''
        x is expected to be (batch, channel, h, w)
        '''
        #print(x.shape)
        x = self.relu(self.conv1(x))
        x = self.relu(self.upconv1(x))
        x = self.relu(self.upconv2(x))
        x = self.relu(self.upconv3(x))
        x = self.relu(self.conv2(x))
        x = self.tanh(self.conv3(x)) # try with tanh and relu and sigmoid
        return x	


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
			#print(f'orig size: {images.shape}')
			images, targets = self.transform(images, targets)
			#print(f'new size: {images.tensors.shape}')
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
				# print(f'polar size: {polars.shape}, img size: {imgs.shape}')
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
		for param in self.fasterRCNN.backbone.backbone.body.conv1.parameters():
			param.requires_grad = True

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
		# print(polars.shape, imgs.shape)
		return self.fasterRCNN(imgs, polars, labels)

model = InputInjection()
print(model)
# print(model)
# x = torch.ones((1,3,640,360))
# p = torch.ones((1,24,16,9))
# y = model(x,p)
# print(y[0])

# given the pretrained nature of the model, I suspect zero initialization will be the best
