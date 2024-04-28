from torch.utils.data import DataLoader
from torchvision import transforms
import transformers
import torch
from PIL import Image
import os
import time
import uuid

from MambaVision.dataset import OpenImagesDataset, OpenImagesDatasetYolo
from MambaVision.utils.metrics import calculate_metrics
import yolov5
from yolov5.models.common import AutoShape


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_DIR = 'imgs'

def display_image(x, output_dir="imgs"):
    if not os.path.exists(IMG_DIR):
        os.makedirs(output_dir)
    if "raw" in output_dir:
        image_id = str(uuid.uuid4())
        for i in range(x.size(0)):
            img = transforms.ToPILImage()(x[i])
            img.save(os.path.join(IMG_DIR, f'image_{image_id}_{i}.jpg'))
    elif "predictions" in output_dir:
        for i in range(x.size(0)):
            x.save(os.path.join(IMG_DIR, f'image_{i}.jpg'))
            # img = Image.fromarray(x[i].mul(255).permute(1, 2, 0).byte().cpu().numpy())
            # img.save(os.path.join(IMG_DIR, f'image_{i}.jpg'))

def test_yolo(yolo, test_dataset, display_raw=False, display_prediction=False):
    for f in os.listdir(IMG_DIR):
        os.remove(os.path.join(IMG_DIR, f))
    yolo.eval()
    yolo.to(DEVICE)

    all_predictions = []
    all_ground_truths = []

    for images, targets in test_dataset:
        with torch.no_grad():
            predictions = yolo(images)
        if display_raw:
            display_image(images, output_dir="imgs/raw")
        all_predictions.append(predictions)
        all_ground_truths.append(targets)
    

    precision, recall, mAP = calculate_metrics(all_predictions, all_ground_truths)
    print(f"Precision: {precision}, Recall: {recall}, mAP: {mAP}")

def test_vit(model, test_dataset):
    for f in os.listdir(IMG_DIR):
        os.remove(os.path.join(IMG_DIR, f))

    all_predictions = []
    all_ground_truths = []

    for images, targets in test_dataset:
        img_shape = images.shape[-2:]
        with torch.no_grad():
            predictions = model(images.to(DEVICE))
            img_w = targets[0].orig_w.item()
            img_h = targets[0].orig_h.item()
            predictions['pred_boxes'] = predictions['pred_boxes'] * torch.tensor([img_w, img_h, img_w, img_h], device=DEVICE)
        all_predictions.append(predictions)
        all_ground_truths.append(targets)

    precision, recall, mAP = calculate_metrics(all_predictions, all_ground_truths)
    print(f"Precision: {precision}, Recall: {recall}, mAP: {mAP}")

def load_model(name, reload_data=False, eval_size=10):
    if name == "yolov5":
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        dataset = OpenImagesDatasetYolo('dataset', ['Car'], download=reload_data, limit=eval_size)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    elif name == "detr":
        model = torch.hub.load("facebookresearch/detr", "detr_resnet50", pretrained=True)
        model.eval()
        model.to(DEVICE)
        dataset = OpenImagesDataset('dataset', ['Car'], download=reload_data, limit=eval_size)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return model, dataloader

if __name__ == "__main__":
    model_name = "yolov5"
    eval_size = 100
    print("Using model: ", model_name)
    model, dataloader = load_model(model_name, reload_data=True, eval_size=eval_size)
    if model_name == "yolov5":
        test_yolo(model, dataloader)
    elif model_name == "detr":
        test_vit(model, dataloader)
