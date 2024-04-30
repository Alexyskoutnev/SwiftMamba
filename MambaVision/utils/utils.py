import torch
import datetime as dt
from MambaVision.utils.metrics import pred_label

NUM_2_CLASSES = ["car", "ambulance", "bicycle", "bus", "helicopter", "motorcycle", "truck", "van"]
CLASSES_2_NUM = {v: k for k, v in enumerate(NUM_2_CLASSES)}
SAVE_MODEL_PATH = "models"

def save_model(model):
    """
    Save the model
    """
    _time = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    torch.save(model.state_dict(), f"{SAVE_MODEL_PATH}/{model.__class__.__name__}_{_time}.pt")
    print(f"Saved to {SAVE_MODEL_PATH}/{model.__class__.__name__}_{_time}.pt")


def load_model(model, path):
    """
    Load the model
    """
    model.load_state_dict(torch.load(path))
    print(f"Loaded from {path}")



def bounding_box_tensor(pred_labels, device=None):
    """
    Convert the predicted labels to a bounding box tensor
    Args:
        pred_label (list): The predicted labels
    Returns:
        torch.Tensor: The bounding box tensor
    """
    img_w, img_h = pred_labels[0].orig_w.item(), pred_labels[0].orig_h.item()
    pred_t = torch.zeros((len(pred_labels), 4), dtype=torch.float32).to(device)
    for i, pred_label in enumerate(pred_labels):
        pred_t[i] = torch.tensor([pred_label.xmin / img_w, pred_label.ymin / img_h, pred_label.xmax / img_w, pred_label.ymax / img_h], dtype=torch.float32)
    return pred_t

def bounding_box_to_labels(bboxs, label, img_w, img_h, device=None):
    """
    Convert the bounding box tensor to labels
    Args:
        bboxs (torch.Tensor): The bounding box tensor
    Returns:
        list: The predicted labels
    """
    pred_labels = []
    for bbox in bboxs:
        xmin, ymin, xmax, ymax = bbox
        xmin = xmin.item()
        ymin = ymin.item()
        xmax = xmax.item()
        ymax = ymax.item()
        pred_labels.append(pred_label(label, xmin, ymin, xmax, ymax, img_h, img_w))
    return pred_labels

def mamba_num_to_class(x):
    """
    Convert the integer to class
    Args:
        x (int): The integer
    Returns:
        str: The class
    """
    dict = {0: "Car", 1: "Ambulance", 2: "Bicycle", 3: "Bus", 4: "Helicopter", 5: "Motorcycle", 6: "Truck", 7: "Van"}
    return dict[x]