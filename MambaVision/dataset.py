import torch
import torchvision.transforms as transforms
from PIL import Image
from openimages.download import download_dataset

import glob
import xml.etree.ElementTree as ET

def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    filename = root.find('filename').text
    objects = []
    for obj in root.findall('object'):
        obj_name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        xmax = int(bbox.find('xmax').text)
        ymin = int(bbox.find('ymin').text)
        ymax = int(bbox.find('ymax').text)
        objects.append({
            'class': obj_name,
            'bbox': (xmin, ymin, xmax, ymax)
        })

    return filename, objects

class OpenImagesDataset(torch.utils.data.Dataset):
    def __init__(self, download_dir, classes_name, transform=None, limit=100, download=False):
        if transform is None:
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor()
                ])
        self.transform = transform
        if download:
            self.dataset = download_dataset(download_dir, classes_name, annotation_format="pascal", limit=limit)
            self.images = glob.glob(f"{download_dir}/{classes_name[0].lower()}/images/*.jpg")
            self.labels = glob.glob(f"{download_dir}/{classes_name[0].lower()}/pascal/*.xml")
        else:
            self.images = glob.glob(f"{download_dir}/{classes_name[0].lower()}/images/*.jpg")
            self.labels = glob.glob(f"{download_dir}/{classes_name[0].lower()}/pascal/*.xml")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        if image.mode != 'RGB':
            image = image.convert('RGB')
        label_path = self.labels[idx]
        label = parse_annotation(label_path)
        if self.transform:
            image = self.transform(image)
        return image, label

class OpenImagesDatasetYolo(torch.utils.data.Dataset):
    def __init__(self, download_dir, classes_name, transform=None, limit=100, download=False):
        if download:
            self.dataset = download_dataset(download_dir, classes_name, annotation_format="pascal", limit=limit)
            self.images = glob.glob(f"{download_dir}/{classes_name[0].lower()}/images/*.jpg")
            self.labels = glob.glob(f"{download_dir}/{classes_name[0].lower()}/pascal/*.xml")
        else:
            self.images = glob.glob(f"{download_dir}/{classes_name[0].lower()}/images/*.jpg")
            self.labels = glob.glob(f"{download_dir}/{classes_name[0].lower()}/pascal/*.xml")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label_path = self.labels[idx]
        label = parse_annotation(label_path)
        return image, label


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
