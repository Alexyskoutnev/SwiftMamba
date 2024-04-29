import torch

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