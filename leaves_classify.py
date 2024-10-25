import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils import data
from PIL import Image
import os
from torchvision import transforms
from torch import nn
from torch.nn import functional as F

# Define a custom dataset class to load images and labels
class ImageDataset(data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None, is_train=True):
        super().__init__()
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir # images root directory
        self.transform = transform # image transformation pipeline
        self.is_train = is_train

        # Create label mapping
        if is_train:
            self.label_mapping = {label: idx for idx, label in enumerate(self.data_frame.iloc[:, 1].unique())}
            self.label_reserve_mapping = {idx: label for label, idx in self.label_mapping.items()}
        else:
            self.label_mapping = None
            self.label_reserve_mapping = None

    def __len__(self):
        # Return the total number of samples
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Retrieve image path and label for the given index
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0]) # 图像名字？
        img_relative_path = self.data_frame.iloc[idx, 0]
        image = Image.open(img_name).convert('RGB') # Ensure 3-channel image (RGB)

        if self.is_train:
            label = self.data_frame.iloc[idx, 1]
            label = self.label_mapping[label] # Map label to int
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.long)
        else:
            if self.transform:
                image = self.transform(image)
            return image, img_relative_path

# Image transformations
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# File paths for training and testing data
train_csv = 'D:\\LearnPytorch\\Kaggle\\Classify Leaves\\leaves_data\\train.csv'
test_csv = 'D:\\LearnPytorch\\Kaggle\\Classify Leaves\\leaves_data\\test.csv'
images_folder = 'D:\\LearnPytorch\\Kaggle\\Classify Leaves\\leaves_data'

# Create training and testing dataset
train_dataset = ImageDataset(csv_file=train_csv, root_dir=images_folder, transform=transform, is_train=True)
test_dataset = ImageDataset(csv_file=test_csv, root_dir=images_folder, transform=transform, is_train=False)

# Create Data loaders
batch_size = 64

train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the BottleNeck block for ResNet
class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, strides=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=strides, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.donwsample = downsample

    def forward(self,X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.donwsample is not None:
            X = self.donwsample(X)
        Y += X
        return F.relu(Y)

# Define ResNet model
class ResNet(nn.Module):
    def __init__(self, block, num_residual, num_classes=176):
        super().__init__()
        self.in_channels = 64
        self.s1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                           nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, 2, padding=1))
        self.s2 = self.resnet_stage(block, 64, num_residual[0])
        self.s3 = self.resnet_stage(block, 128, num_residual[1], stride=2)
        self.s4 = self.resnet_stage(block, 256, num_residual[2], stride=2)
        self.s5 = self.resnet_stage(block, 512, num_residual[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def resnet_stage(self, block, out_channels, num_residual, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels * block.expansion,
                                                 kernel_size=1, stride=stride),
                                       nn.BatchNorm2d(out_channels * block.expansion))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_residual):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Define ResNet50 model
def ResNet50(num_classes=176):
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes)

# Device, model, and training parameters
device = torch.device('cuda')
model = ResNet50().to(device)
loss = nn.CrossEntropyLoss()
num_epochs, lr, wd = 20, 0.0005, 1e-4
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=wd)

# Initialize weights using Kaiming initialization
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.kaiming_uniform_(m.weight)

model.apply(init_weights)

# Training loop
train_loss = []
train_accuracies, test_accuracies = [], []

for epochs in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_total, train_correct = 0, 0
    test_total, test_correct = 0, 0

    for X_train, y_train in train_loader:

        X_train, y_train = X_train.to(device), y_train.to(device)
        l = loss(model(X_train), y_train)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        running_loss += l.item() * X_train.size(0)
        _, train_predictor = torch.max(model(X_train).data, 1)  # Get the index of the max
        train_total += y_train.size(0)
        train_correct += (train_predictor == y_train).sum().item()
    avg_loss = running_loss / len(train_loader.dataset)
    train_accuracy = 100 * train_correct / train_total
    train_accuracies.append(train_accuracy)
    train_loss.append(avg_loss)
    print(f"Epoch [{epochs+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.2f}%")


    # Testing on test data
    all_labels = []
    image_filenames = []

    with torch.no_grad():
        model.eval()
        for X_test, image_filename in test_loader:  # Returns both image and filename
            X_test = X_test.to(device)

            # Forward pass and get the model output
            labels = model(X_test).detach().cpu().numpy()
            labels = np.argmax(labels, axis=1)  # Convert model output (logits) to class predictions

            # Collect predicted labels and image filenames
            all_labels.extend(labels)
            image_filenames.extend(image_filename)
    # Map predicted labels back to original label names
    all_class_labels = [train_dataset.label_reserve_mapping[label] for label in all_labels]

    # Ensure that the number of predictions matches the number of image filenames
    assert len(all_labels) == len(image_filenames), "Mismatch between number of images and predictions"

    # Create the submission DataFrame
    submission = pd.DataFrame({
        'image': image_filenames,  # Image filenames (e.g., '18353.jpg')
        'label': all_class_labels  # Predicted labels
    })

    # Save the submission file
    submission.to_csv('submission.csv', index=False)

# Plot training loss and accuracy
plt.figure(figsize=(16, 12))
plt.plot(train_loss, label='Train Loss', linestyle='-')
plt.plot(train_accuracies, label='Train Acc', marker='x', color='b')
plt.xlabel('Epoch')
plt.ylabel('Percentage / Loss')
plt.legend()
plt.show()