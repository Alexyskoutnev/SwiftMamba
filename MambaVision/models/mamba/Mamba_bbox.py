from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models import create_model
import argparse

from MambaVision.utils.utils import *

class BBoxHead(nn.Module):
    def __init__(self, in_features, num_outputs):
        super(BBoxHead, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, num_outputs)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class BBoxClassificationHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super(BBoxClassificationHead, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class VisionMambaBBox(nn.Module):
    def __init__(self, 
                 base_model,
                 num_classes=100):
        super().__init__()
        self.mamba = base_model
        self.mamba.head = torch.nn.Identity()
        self.class_head = BBoxClassificationHead(self.mamba.norm.bias.shape[0], num_classes)
        self.bc_head = BBoxHead(self.mamba.norm.bias.shape[0], 4)

    def forward(self, x):
        x = self.mamba(x)
        class_predictions = self.class_head(x) # [num_classes]
        bbox_predictions = self.bc_head(x) # [w_left_top, h_left_top, width, height]
        return class_predictions, bbox_predictions


class BBoxLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss()
        self.class_loss_fn = nn.CrossEntropyLoss()

    def forward(self, pred_bbox : List[int], target_bbox : List[int], pred_class : int, target_class: int, device: str):
        pred_box = box_cxyh_to_xyxy(pred_bbox)
        bbox_loss = self.loss_fn(pred_bbox, target_bbox)
        class_loss = self.class_loss_fn(pred_class, target_class)
        return bbox_loss + class_loss