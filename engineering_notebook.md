# Engineering Notebook

### May 6, RPN-Injection Model

The current RPN model architecture slightly modifies the backbone of the faster rcnn:

```python
polar_backbone_pre: a conv layer from 24 to 96
bilinear interpolation to correct sizes of the outputs of fpn
polar_backbone_post: a list of conv layers, separately for each fpn output, from 96 to 256

img_backbone: resnet50+fpn
```

The training result of the layer is terrible --> showing a decline in accuracy as training progresses. The loss jumps a lot, indicating a high learning rate. Further, the requires_grad parameters is probably set correctly but may not be the case.

I've updated the requires_grad parameter setting to allow freezing the entire img_backbone while allowing the polar_backbones to be trained.  

Another possibility is that the spectrogram doesn't match the image---a fault of data loading, I will check shortly after.

Also, we must consider the fact that $96*256*3*3$ is relatively a large number of parameters. Perhaps the transition from 24 to 96 can be made in one singular stage using one singular layer, so as to limit the hypothesis space of the model.

Further examination of the log output shows that the loss declines from 0.2 to 0.01 over the course of 100 epochs of training. This is a clear indication of overfitting---intra-video images are so similar that the model has opted to learn the videos not the images separately. The main ways for resolving this is: 1. reducing the trainability of the model; 2. appending an adversarial head

Starting from log-may-7-c, I will reduce the complexity of the model. After that I will limit the trainability by freezing more layers (even maybe intermediate ones). 

### May 7, RPN-Injection Model

Starting from log-may-7-c, I not only reduced the complexity of the polarization head but also restructured the dataset to make sure that the dataset is correct. The new AP50 is around 0.52 and doesn't decrease as dramatically. Nonetheless, we see the AP value saturate even as our loss continues to decrease. The trainability of the model is still too large. In particular, I'm talking about the feature processing layers before the RPN, and the processing layers in the ROI heads. Perhaps the injection should happen after those layers. In the meantime, I'm curious to see how well the model would do if we just finetune the last layer and don't add features (the vanilla faster rcnn)