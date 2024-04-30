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

class VisionMambaWithBBox(nn.Module):
    def __init__(self, 
                 base_model,):
        super().__init__()
        self.mamba = base_model
        self.mamba.head = torch.nn.Identity()
        self.bc_head = BBoxHead(768, 4)

    def forward(self, x):
        x = self.mamba(x)
        return self.bc_head(x)

# Define the loss function for bounding box prediction
class BBoxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss()

    def forward(self, pred_bbox, target_bbox):
        return self.loss_fn(pred_bbox, target_bbox)