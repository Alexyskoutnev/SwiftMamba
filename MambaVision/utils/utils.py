from typing import Tuple

import torch
import datetime as dt
from MambaVision.utils.metrics import pred_label

NUM_2_CLASSES = ["car", "ambulance", "bicycle", "bus", "helicopter", "motorcycle", "truck", "van"]
CLASSES_2_NUM = {v: k for k, v in enumerate(NUM_2_CLASSES)}
SAVE_MODEL_PATH = "models"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_model(model):
    """
    Save the model
    """
    _time = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    torch.save(model.state_dict(), f"{SAVE_MODEL_PATH}/{model.__class__.__name__}_{_time}.pt")
    print(f"Saved to {SAVE_MODEL_PATH}/{model.__class__.__name__}_{_time}.pt")


def load(model, path):
    """
    Load the model
    """
    model.load_state_dict(torch.load(path))
    print(f"Loaded from {path}")
    return model

def single_bounding_box_label(box, label, img_h, img_w):
    """
    Convert the single bounding box to label
    Args:
        label (int): The label
        box (list): The bounding box
        img_h (int): The image height
        img_w (int): The image width
    Returns:
        dict: The label
    """
    return pred_label(NUM_2_CLASSES[label], box[0][0].item(), box[0][1].item(), box[0][2].item(), box[0][3].item(), img_h, img_w)

def bounding_box_tensor(element, device=None):
    """
    Convert the predicted labels to a bounding box tensor
    Args:
        pred_label (list): The predicted labels
    Returns:
        torch.Tensor: The bounding box tensor
    """
    pred_t = torch.zeros((len(element[0][5]), 4), dtype=torch.float32).to(device)
    labels = element[0][0]
    xmin = element[0][1].tolist()
    ymin = element[0][2].tolist()
    xmax = element[0][3].tolist()
    ymax = element[0][4].tolist()
    img_h = element[0][5].tolist()
    img_w = element[0][6].tolist()
    for i in range(len(labels)):
        pred_t[i] = torch.tensor([xmin[i] / img_w[i], ymin[i] / img_h[i], xmax[i] / img_w[i], ymax[i] / img_h[i]], dtype=torch.float32)
    return pred_t.to(device)

def bounding_box_to_labels(bboxs, label, img_w, img_h, device=None):
    """
    Convert the bounding box tensor to labels
    Args:
        bboxs (torch.Tensor): The bounding box tensor
    Returns:
        list: The predicted labels
    """
    pred_labels = []
    for bbox, label, w, h in zip(bboxs, label, img_w, img_h):
        xmin, ymin, xmax, ymax = bbox
        xmin = xmin.item()
        ymin = ymin.item()
        xmax = xmax.item()
        ymax = ymax.item()
        pred_labels.append(pred_label(label, xmin, ymin, xmax, ymax, h, w))
    return pred_labels

def mamba_num_to_class(x : torch.Tensor):
    """
    Convert the integer to class
    Args:
        x (int): The integer
    Returns:
        str: The class
    """
    dict = {0: "Car", 1: "Ambulance", 2: "Bicycle", 3: "Bus", 4: "Helicopter", 5: "Motorcycle", 6: "Truck", 7: "Van"}
    if type(x) == int:
        return dict[x]
    labels = []
    for label in x:
        labels.append(dict[label.item()])
    return labels

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def box_cxyh_to_xyxy(b):
    x_min, y_min, w, h= b.unbind(1)
    x_max = x_min + w
    y_max = y_min + h
    return torch.stack((x_min, y_min, x_max, y_max), dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    img_w = torch.tensor([img_w], dtype=torch.float32, device=DEVICE)
    img_h = torch.tensor([img_h], dtype=torch.float32, device=DEVICE)
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], device=DEVICE)
    return b

def filter_bboxes_from_outputs(outputs, size, threshold=0.75):
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], size)
    return probas[keep], bboxes_scaled

def target_human_to_tensor_label(labels : Tuple[str]):
    """
    Convert the target labels to tensor labels
    Args:
        labels (Tuple[str]): The target labels
    Returns:
        torch.Tensor: The tensor labels
    """
    labels = [CLASSES_2_NUM[label] for label in labels]
    return torch.tensor(labels, dtype=torch.long).to(DEVICE)