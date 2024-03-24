import torch
import torchvision.transforms as transforms
from PIL import Image
from openimages.download import download_dataset

class OpenImagesDataset(torch.utils.data.Dataset):
    def __init__(self, download_dir, classes_name, transform=None, limit=100):
        self.transform = transform
        self.images, self.labels = download_dataset(download_dir, classes_name, annotation_format="pascal", limit=limit)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == "__main__":
    classes = ['Person', 'Car', 'Dog']
    download_dir = 'dataset'
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset = OpenImagesDataset(download_dir, classes, transform=transform)
    breakpoint()
