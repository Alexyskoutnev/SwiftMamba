from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from PIL import Image
import os
import time
import uuid

from MambaVision.dataset import OpenImagesDataset, OpenImagesDatasetYolo
from MambaVision.utils.metrics import calculate_metrics
import yolov5
from yolov5.models.common import AutoShape

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo.to(device)

    all_predictions = []
    all_ground_truths = []

    for images, targets in test_dataset:  # Corrected variable name from 'image' to 'images'
        with torch.no_grad():
            predictions = yolo(images)  # Move images to device before passing to model
        if display_raw:
            display_image(images, output_dir="imgs/raw")
        # if display_prediction:
        #     display_image(predictions, output_dir="imgs/predictions")
        all_predictions.append(predictions)
        all_ground_truths.append(targets)
    
    # Calculate precision, recall, and mAP
    precision, recall, mAP = calculate_metrics(all_predictions, all_ground_truths)
    print(f"Precision: {precision}, Recall: {recall}, mAP: {mAP}")


if __name__ == "__main__":
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    # target_classes = ['car']
    # model.classes = target_classes
    test_dataset = OpenImagesDatasetYolo('dataset', ['Car'], download=False, limit=10)
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_yolo(model, dataloader, display_raw=False, display_prediction=True)
