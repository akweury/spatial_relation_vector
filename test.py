# Created by shaji at 02.12.2022

import torchvision
import pycocotools

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

object_detection_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=False)
object_detection_model.eval()

