import torch
import os
import torchvision.transforms as transforms
from PIL import Image
from openimages.download import download_dataset

import glob
import xml.etree.ElementTree as ET
from MambaVision.utils.metrics import pred_label

def preprocess_dataset_one_label(files):
    files_with_single_car = []
    deleted_labels = []
    for file in files:
        tree = ET.parse(file)
        root = tree.getroot()
        num_objs = len(root.findall('object'))
        if num_objs == 1:
            files_with_single_car.append(file)
        else:
            deleted_labels.append(file)
    return files_with_single_car, deleted_labels

def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    filename = root.find('filename').text
    objects = []
    for obj in root.findall('size'):
        width = int(obj.find('width').text)
        height = int(obj.find('height').text)
    for obj in root.findall('object'):
        obj_name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        xmax = int(bbox.find('xmax').text)
        ymin = int(bbox.find('ymin').text)
        ymax = int(bbox.find('ymax').text)
        obj = pred_label(obj_name, xmin, ymin, xmax, ymax, height, width)
    objects.append(obj)
    return objects

class OpenImagesDataset(torch.utils.data.Dataset):
    def __init__(self, download_dir, classes_name, transform=None, limit=100, download=False):
        if transform is None:
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor()
                ])
                
        self.transform = transform
        self.transform2 = transforms.Compose([
            transforms.ToTensor()
        ])
        if download:
            if not os.path.exists(download_dir):
                os.makedirs(download_dir)
            if os.path.exists(f"{download_dir}/{classes_name[0].lower()}"):
                os.system(f"rm -rf {download_dir}/{classes_name[0].lower()}")
            self.dataset = download_dataset(download_dir, classes_name, annotation_format="pascal", limit=limit)
            self.images = glob.glob(f"{download_dir}/{classes_name[0].lower()}/images/*.jpg")
            self.labels, deleted_labels =  preprocess_dataset_one_label(glob.glob(f"{download_dir}/{classes_name[0].lower()}/pascal/*.xml"))
            for label in deleted_labels:
                file_name = label.split('/')[-1]
                file_name = file_name.split('.')[0]
                file_name += '.jpg'
                self.images.remove(f"{download_dir}/{classes_name[0].lower()}/images/{file_name}")
        else:
            self.images = glob.glob(f"{download_dir}/{classes_name[0].lower()}/images/*.jpg")
            self.images = sorted(self.images)
            self.labels = glob.glob(f"{download_dir}/{classes_name[0].lower()}/pascal/*.xml")
            self.labels = sorted(self.labels)
            self.labels, deleted_labels = preprocess_dataset_one_label(self.labels)
            for label in deleted_labels:
                file_name = label.split('/')[-1]
                file_name = file_name.split('.')[0]
                file_name += '.jpg'
                self.images.remove(f"{download_dir}/{classes_name[0].lower()}/images/{file_name}")
                print("Deleted: ", f"{download_dir}/{classes_name[0].lower()}/images/{file_name}")

    def __len__(self):
        return len(self.images)
    
    def _valiate(self, path1, path2):
        file_name1 = path1.split('/')[-1]
        file_name1 = file_name1.split('.')[0]
        file_name2 = path2.split('/')[-1]
        file_name2 = file_name2.split('.')[0]
        if file_name1 != file_name2:
            raise ValueError("File names are not same")

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        orignal_img = image.copy()
        if image.mode != 'RGB':
            image = image.convert('RGB')
        label_path = self.labels[idx]
        label = parse_annotation(label_path)
        if self.transform:
            image = self.transform(image)
        if self.transform2:
            orignal_img = self.transform2(orignal_img)
        self._valiate(self.images[idx], self.labels[idx])
        return image, label, orignal_img

class OpenImagesDatasetYolo(torch.utils.data.Dataset):
    def __init__(self, download_dir, classes_name, transform=None, limit=100, download=False):
        if download:
            if not os.path.exists(download_dir):
                os.makedirs(download_dir)
            if os.path.exists(f"{download_dir}/{classes_name[0].lower()}"):
                os.system(f"rm -rf {download_dir}/{classes_name[0].lower()}")
            self.dataset = download_dataset(download_dir, classes_name, annotation_format="pascal", limit=limit)
            self.images = glob.glob(f"{download_dir}/{classes_name[0].lower()}/images/*.jpg")
            self.labels, deleted_labels =  preprocess_dataset_one_label(glob.glob(f"{download_dir}/{classes_name[0].lower()}/pascal/*.xml"))
            for label in deleted_labels:
                file_name = label.split('/')[-1]
                file_name = file_name.split('.')[0]
                file_name += '.jpg'
                self.images.remove(f"{download_dir}/{classes_name[0].lower()}/images/{file_name}")
        else:
            self.images = glob.glob(f"{download_dir}/{classes_name[0].lower()}/images/*.jpg")
            self.images = sorted(self.images)
            self.labels = glob.glob(f"{download_dir}/{classes_name[0].lower()}/pascal/*.xml")
            self.labels = sorted(self.labels)
            self.labels, deleted_labels = preprocess_dataset_one_label(self.labels)
            for label in deleted_labels:
                file_name = label.split('/')[-1]
                file_name = file_name.split('.')[0]
                file_name += '.jpg'
                self.images.remove(f"{download_dir}/{classes_name[0].lower()}/images/{file_name}")
                print("Deleted: ", f"{download_dir}/{classes_name[0].lower()}/images/{file_name}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        assert len(self.images) == len(self.labels)
        image = self.images[idx]
        label_path = self.labels[idx]
        label = parse_annotation(label_path)
        self._valiate(self.images[idx], self.labels[idx])
        return image, label
    
    def _valiate(self, path1, path2):
        file_name1 = path1.split('/')[-1]
        file_name1 = file_name1.split('.')[0]
        file_name2 = path2.split('/')[-1]
        file_name2 = file_name2.split('.')[0]
        if file_name1 != file_name2:
            raise ValueError("File names are not same")


if __name__ == "__main__":
    classes = ['Car']
    download_dir = 'dataset'
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    limit = 100 
    dataset = OpenImagesDataset(download_dir, classes, transform=transform, limit=limit)
    for image, label in dataset:
        print(image, label)
