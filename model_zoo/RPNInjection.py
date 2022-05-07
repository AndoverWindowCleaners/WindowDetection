import warnings
from typing import Tuple, List, OrderedDict, Dict

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import misc as misc_nn_ops

from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor

from torchvision.models.resnet import resnet50
from torchvision._internally_replaced_utils import load_state_dict_from_url
from torchvision.models.detection._utils import overwrite_eps

model_urls = {
	"fasterrcnn_resnet50_fpn_coco": "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
}

class PolarBackbonePre(nn.Module):
	def __init__(self, in_channels) -> None:
		super().__init__()
		self.conv = nn.Conv2d(in_channels, 96, (5, 5), padding='same')
	def forward(self, x) -> Tensor:
		return self.conv(x)

class PolarBackbonePost(nn.Module):
	def __init__(self, out_channels, input_levels=5) -> None:
		super().__init__()
		self.convs = nn.ModuleList()
		for i in range(input_levels):
			self.convs.append(nn.Conv2d(96, out_channels, (3, 3), padding='same'))
	def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
		return {key : self.convs[i](val) for i,(key,val) in enumerate(x.items())}

class BackboneWithPolarizer(nn.Module):
	def __init__(
		self,
		img_backbone: nn.Module,
		polar_in_channels = 24,
	) -> None:
		super().__init__()
		self.out_channels = img_backbone.out_channels
		self.img_backbone = img_backbone
		self.polar_backbone_pre = PolarBackbonePre(polar_in_channels)
		self.polar_backbone_post = PolarBackbonePost(self.out_channels)

	def freeze_img_backbone(self):
		for param in self.parameters():
			param.requires_grad = True
		for param in self.img_backbone.parameters():
			param.requires_grad = False

	def forward(self, img: Tensor, polar: Tensor) -> Dict[str, Tensor]:
		imgs = self.img_backbone(img)
		polar = self.polar_backbone_pre(polar)
		polars = {key : F.interpolate(polar, (val.shape[-2], val.shape[-1]), mode='bilinear') for key,val in imgs.items()}
		polars = self.polar_backbone_post(polars)
		features = {key : imgs[key]+polars[key] for key in imgs.keys()}
		return features

class RPNInjection(GeneralizedRCNN): # see FasterRCNN's source code
	def __init__(
		self,
		backbone,
		num_classes=None,
		# transform parameters
		min_size=800,
		max_size=1333,
		image_mean=None,
		image_std=None,
		# RPN parameters
		rpn_anchor_generator=None,
		rpn_head=None,
		rpn_pre_nms_top_n_train=2000,
		rpn_pre_nms_top_n_test=1000,
		rpn_post_nms_top_n_train=2000,
		rpn_post_nms_top_n_test=1000,
		rpn_nms_thresh=0.7,
		rpn_fg_iou_thresh=0.7,
		rpn_bg_iou_thresh=0.3,
		rpn_batch_size_per_image=256,
		rpn_positive_fraction=0.5,
		rpn_score_thresh=0.0,
		# Box parameters
		box_roi_pool=None,
		box_head=None,
		box_predictor=None,
		box_score_thresh=0.05,
		box_nms_thresh=0.5,
		box_detections_per_img=100,
		box_fg_iou_thresh=0.5,
		box_bg_iou_thresh=0.5,
		box_batch_size_per_image=512,
		box_positive_fraction=0.25,
		bbox_reg_weights=None,
	):

		if not hasattr(backbone, "out_channels"):
			raise ValueError(
				"backbone should contain an attribute out_channels "
				"specifying the number of output channels (assumed to be the "
				"same for all the levels)"
			)

		assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
		assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

		if num_classes is not None:
			if box_predictor is not None:
				raise ValueError("num_classes should be None when box_predictor is specified")
		else:
			if box_predictor is None:
				raise ValueError("num_classes should not be None when box_predictor is not specified")

		out_channels = backbone.out_channels

		if rpn_anchor_generator is None:
			anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
			aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
			rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
		if rpn_head is None:
			rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])

		rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
		rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

		rpn = RegionProposalNetwork(
			rpn_anchor_generator,
			rpn_head,
			rpn_fg_iou_thresh,
			rpn_bg_iou_thresh,
			rpn_batch_size_per_image,
			rpn_positive_fraction,
			rpn_pre_nms_top_n,
			rpn_post_nms_top_n,
			rpn_nms_thresh,
			score_thresh=rpn_score_thresh,
		)

		if box_roi_pool is None:
			box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

		if box_head is None:
			resolution = box_roi_pool.output_size[0]
			representation_size = 1024
			box_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)

		if box_predictor is None:
			representation_size = 1024
			box_predictor = FastRCNNPredictor(representation_size, num_classes)

		roi_heads = RoIHeads(
			# Box
			box_roi_pool,
			box_head,
			box_predictor,
			box_fg_iou_thresh,
			box_bg_iou_thresh,
			box_batch_size_per_image,
			box_positive_fraction,
			bbox_reg_weights,
			box_score_thresh,
			box_nms_thresh,
			box_detections_per_img,
		)

		if image_mean is None:
			image_mean = [0.485, 0.456, 0.406]
		if image_std is None:
			image_std = [0.229, 0.224, 0.225]
		transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

		super().__init__(backbone, rpn, roi_heads, transform)


	def load_limited_state_dict(self, state_dict):
		own_state = self.state_dict()
		for name, param in state_dict.items():
			if name.startswith('backbone.') :
				name = 'backbone.img_backbone.'+name[len('backbone.'):]
			if name not in own_state:
				print(f'skip: {name}')
				continue
			if name.startswith('roi_heads.box_predictor'):
				continue
			print(f'load: {name}')
			if isinstance(param, Parameter): # backwards compatibility for serialized parameters
				param = param.data
			own_state[name].copy_(param)

	def freeze_body(self):
		for param in self.parameters():
			param.requires_grad = False
		for param in self.roi_heads.box_predictor.parameters():
			param.requires_grad = True

	def check_box_degeneracy(self, targets):
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
						f" Found invalid box {degen_bb} for target at index {target_idx}."
					)

	def forward(self, images, polars, targets=None):
		"""
		Args:
			images (list[Tensor]): images to be processed
			polars (list[Tensor]): images to be processed
			targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
		Returns:
			result (list[BoxList] or dict[Tensor]): the output from the model.
				During training, it returns a dict[Tensor] which contains the losses.
				During testing, it returns list[BoxList] contains additional fields
				like `scores`, `labels` and `mask` (for Mask R-CNN models).
		"""
		# Preliminary transforms
		if self.training and targets is None:
			raise ValueError("In training mode, targets should be passed")
		if self.training:
			assert targets is not None
			for target in targets:
				boxes = target["boxes"]
				if isinstance(boxes, torch.Tensor):
					if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
						raise ValueError(f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.")
				else:
					raise ValueError(f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

		original_image_sizes: List[Tuple[int, int]] = []
		for img in images:
			val = img.shape[-2:]
			assert len(val) == 2
			original_image_sizes.append((val[0], val[1]))

		images, targets = self.transform(images, targets)
		# Check for degenerate boxes
		self.check_box_degeneracy(targets)

		if isinstance(polars, List):
			polars = torch.stack(polars, axis = 0)
		assert(isinstance(polars, Tensor))

		features = self.backbone(images.tensors, polars)
		if isinstance(features, torch.Tensor):
			features = OrderedDict([("0", features)])
		# structure is always in fpn structure: Dict[id : tensor]


		proposals, proposal_losses = self.rpn(images, features, targets)
		detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
		detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

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


def build_rpn_injection_model(
	pretrained=False, progress=True, num_classes=2, pretrained_backbone=True, trainable_backbone_layers=None, **kwargs
):
	trainable_backbone_layers = _validate_trainable_layers(
		pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3
	)

	if pretrained:
		pretrained_backbone = False

	backbone = resnet50(pretrained=pretrained_backbone, progress=progress, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
	backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
	backbone = BackboneWithPolarizer(backbone)
	model = RPNInjection(backbone, num_classes, **kwargs)
	if pretrained:
		state_dict = load_state_dict_from_url(model_urls["fasterrcnn_resnet50_fpn_coco"], progress=progress)
		model.load_limited_state_dict(state_dict)
		overwrite_eps(model, 0.0)
	for param in model.parameters():
		param.requires_grad = True
	if pretrained or pretrained_backbone:
		model.backbone.freeze_img_backbone()
	return model

# model = build_rpn_injection_model(pretrained=True)
# x = torch.ones((1,3,640,360))
# p = torch.ones((1,24,128,72))
# model.eval()
# y = model(x,p)


# create a separate FPN for the polarization info
# another approach is to attach the polarization info directly to the input data
# or something else we can do is to attach it to the ROI pooling layer 