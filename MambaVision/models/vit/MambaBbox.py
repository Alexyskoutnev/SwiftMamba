import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class BBoxHead(nn.Module):
    def __init__(self, in_features, num_outputs):
        super(BBoxHead, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, num_outputs)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Modify the VisionMamba model
class VisionMambaWithBBox(nn.Module):
    def __init__(self, ...):  # Add necessary parameters for the VisionMamba model
        super().__init__()
        # Initialize VisionMamba layers
        
        # Define bounding box prediction head
        self.bbox_head = BBoxHead(in_features=..., num_outputs=4)  # Adjust input features according to the model architecture

    def forward(self, x):
        # Forward pass through VisionMamba layers
        features = self.forward_features(x)
        
        # Bounding box prediction head
        bbox_pred = self.bbox_head(features)
        
        return bbox_pred

# Define the loss function for bounding box prediction
class BBoxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss()

    def forward(self, pred_bbox, target_bbox):
        return self.loss_fn(pred_bbox, target_bbox)

# Initialize the model, loss function, and optimizer
model = VisionMambaWithBBox(...)
criterion = BBoxLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        optimizer.zero_grad()
        bbox_targets = targets['bbox']  # Assuming 'bbox' contains the ground truth bounding box coordinates
        outputs = model(images)
        loss = criterion(outputs, bbox_targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
