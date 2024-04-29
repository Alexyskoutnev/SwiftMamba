from torch.utils.data import DataLoader
from torchvision import transforms
import transformers
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from timm.models import create_model
from timm.models.vision_transformer import VisionTransformer, _cfg
from MambaVision.models.mamba.models_mamba import VisionMamba
import os
import time
import uuid

from MambaVision.dataset import OpenImagesDataset, OpenImagesDatasetYolo
from MambaVision.utils.utils import bounding_box_tensor
from MambaVision.models.mamba.Mamba_bbox import VisionMambaWithBBox, BBoxLoss
from MambaVision.utils.metrics import calculate_metrics, preprocess_yolo_labels, preprocess_vit_output, preprocess_yolo_output
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

def display_bounding_box_image(images, all_ground_truths, all_predictions):
    for i, (image, ground_truth, prediction) in enumerate(zip(images, all_ground_truths, all_predictions)):
        try: 
            if isinstance(image, torch.Tensor):
                image = image.squeeze(0)
                img = Image.fromarray(image.mul(255).permute(1, 2, 0).byte().cpu().numpy())
            elif isinstance(image, tuple):
                img = image[0]
                img = Image.open(img)
            draw = ImageDraw.Draw(img)
            for pred_label in ground_truth:
                xmin = pred_label.xmin.item()
                ymin = pred_label.ymin.item()
                xmax = pred_label.xmax.item()
                ymax = pred_label.ymax.item()
                box = [xmin, ymin, xmax, ymax]
                label_text = str(pred_label.label_class[0])
                draw.text((xmin + 10, ymin + 10), label_text, fill='green')
                draw.rectangle([box[0], box[1], box[2], box[3]], outline='green')
            for pred_label in prediction:
                try:
                    if pred_label.label_class == 'car':
                        xmin = pred_label.xmin
                        ymin = pred_label.ymin
                        xmax = pred_label.xmax
                        ymax = pred_label.ymax
                        box = [xmin, ymin, xmax, ymax]
                        label_text = str(pred_label.label_class)
                        draw.text((xmin + 10, ymin + 10), label_text, fill='red')
                        draw.rectangle([box[0], box[1], box[2], box[3]], outline='red')
                except:
                    continue
            img.save(os.path.join(IMG_DIR, f'image_{i}.jpg'))
        except:
            continue

def test_yolo(yolo, test_dataset, save_predicted_img=False):
    for f in os.listdir(IMG_DIR):
        os.remove(os.path.join(IMG_DIR, f))
    yolo.eval()
    yolo.to(DEVICE)

    all_predictions = []
    all_ground_truths = []
    images_list = []

    for images, targets in test_dataset:
        with torch.no_grad():
            predictions = yolo(images)

        all_predictions.append(predictions)
        all_ground_truths.append(targets)
        images_list.append(images)
    

    precision, recall, mAP = calculate_metrics(all_predictions, all_ground_truths)
    print(f"Precision: {precision}, Recall: {recall}, mAP: {mAP}")

    if save_predicted_img:
        all_ground_truths_post = preprocess_yolo_labels(all_ground_truths)
        all_predictions_post = preprocess_yolo_output(all_predictions, all_ground_truths_post)
        display_bounding_box_image(images_list, all_ground_truths, all_predictions_post)

def test_vit(model, test_dataset, save_predicted_img=False):
    for f in os.listdir(IMG_DIR):
        os.remove(os.path.join(IMG_DIR, f))

    all_predictions = []
    all_ground_truths = []
    images_list = []

    for images, targets, original_img in test_dataset:
        with torch.no_grad():
            predictions = model(images.to(DEVICE))
            img_w = targets[0].orig_w.item()
            img_h = targets[0].orig_h.item()
            predictions['pred_boxes'] = predictions['pred_boxes'] * torch.tensor([img_w, img_h, img_w, img_h], device=DEVICE)
        all_predictions.append(predictions)
        all_ground_truths.append(targets)
        images_list.append(original_img)

    precision, recall, mAP = calculate_metrics(all_predictions, all_ground_truths)
    print(f"Precision: {precision}, Recall: {recall}, mAP: {mAP}")

    if save_predicted_img:
        all_predictions_post = preprocess_vit_output(all_predictions, preprocess_yolo_labels(all_ground_truths)) 
        display_bounding_box_image(images_list, all_ground_truths, all_predictions_post)

def train_mamba(model, test_dataset, config=None):  

    for f in os.listdir(IMG_DIR):
        os.remove(os.path.join(IMG_DIR, f))

    all_predictions = []
    all_ground_truths = []
    images_list = []
    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(10):
        for images, targets, original_img in test_dataset:
            optimizer.zero_grad()
            predictions = model(images.to(DEVICE))
            gt_t = bounding_box_tensor(targets, device=DEVICE) 
            loss = BBoxLoss()(predictions, gt_t)
            print(f"Loss: {loss.item()}")
            breakpoint()
            loss.backward()
            optimizer.step()



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
    elif name == 'mamba_train':
        model = create_model(
            'deit_base_patch16_224',
            pretrained=False,
            num_classes=100,
            drop_rate=0.0,
            drop_path_rate=0.1,
            drop_block_rate=None,
            img_size=256,
        )
        model = VisionMambaWithBBox(base_model = model)
        model.to(DEVICE)
        dataset = OpenImagesDataset('dataset', ['Car'], download=reload_data, limit=eval_size)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    return model, dataloader

if __name__ == "__main__":
    model_name = "mamba_train"
    eval_size = 100
    print("Using model: ", model_name)
    model, dataloader = load_model(model_name, reload_data=False, eval_size=eval_size)
    if model_name == "yolov5":
        test_yolo(model, dataloader, save_predicted_img=True)
    elif model_name == "detr":
        test_vit(model, dataloader, save_predicted_img=True)
    elif model_name == "mamba_train":
        train_mamba(model, dataloader)
