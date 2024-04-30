from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models import create_model
import argparse

from MambaVision.models.mamba.models_mamba import VisionMamba, vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2
from MambaVision.dataset import OpenImagesDataset
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


class BBoxHead(nn.Module):
    def __init__(self, in_features, num_outputs):
        super(BBoxHead, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, num_outputs)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class VisionMambaBBox(nn.Module):
    def __init__(self, 
                 base_model,
                 num_classes=100):
        super().__init__()
        self.mamba = base_model
        self.mamba.head = torch.nn.Identity()
        self.class_head = nn.Linear(self.mamba.norm.bias.shape[0], num_classes)
        self.bc_head = BBoxHead(self.mamba.norm.bias.shape[0], 4)

    def forward(self, x):
        x = self.mamba(x)
        class_predictions = self.class_head(x)
        bbox_predictions = self.bc_head(x)
        return class_predictions, bbox_predictions


class BBoxLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss()
        self.class_loss_fn = nn.CrossEntropyLoss()

    def forward(self, pred_bbox : List[int], target_bbox : List[int], pred_class : int, target_class: int, device: str):
        bbox_loss = self.loss_fn(pred_bbox, target_bbox)
        target_class = torch.tensor([target_class], dtype=torch.long).to(device)
        class_loss = self.class_loss_fn(pred_class, target_class)
        return bbox_loss + class_loss