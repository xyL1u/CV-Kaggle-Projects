import numpy as np
import torch
import os
import pandas as pd
from torch import nn
import torchvision
from torchvision.transforms import transforms
from torch.utils import data
from PIL import Image
from torch.cuda.amp import autocast, GradScaler

# Define a custom dataset class for handling CIFAR-10 dataset
class MyData(data.Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        super().__init__()
        # Load CSV file containing image labels
        self.csv_file = pd.read_csv('D:\\LearnPytorch\\Kaggle\\cifar10\\trainLabels.csv')
        self.root_dir = root_dir # Root directory where images are stored
        self.transform = transform # Images transformations
        self.is_train = is_train

        if is_train:
            # For training, store image-label pairs and create label mappings. Since the labels are str and map them into index
            self.image_labels = list(self.csv_file.itertuples(index=False, name=None))
            self.label_mapping = {label: idx for idx, label in enumerate(self.csv_file.iloc[:, 1].unique())}
            self.reverse_mapping = {idx: label for label, idx in self.label_mapping.items()}
        else:
            # For testing, list all test images
            self.test_files = os.listdir(root_dir)

    def __len__(self):
        # Return the total number of samples in the dataset
        if self.is_train:
            return len(self.image_labels)
        else:
            return len(self.test_files)

    def __getitem__(self, idx):
        # Retrieve image and label by index
        if self.is_train:
            file_name, label = self.image_labels[idx]
            image_path = os.path.join(self.root_dir, f'{file_name}.png')
            image = Image.open(image_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            label = self.label_mapping[label]

            return image, torch.tensor(label, dtype=torch.long)

        else:
            # For test, retrieve only the image and image name
            file_name = self.test_files[idx]
            image_path = os.path.join(self.root_dir, file_name)
            image = Image.open(image_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            return image, file_name
# Define image transformations
train_transform = transforms.Compose([transforms.Resize(40),
                                      transforms.RandomResizedCrop(size=32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])])
# Training and testing image root directories
train_root_dir = 'train/train'
test_root_dir = 'test/test'

# Create dataset instances
train_dataset = MyData(train_root_dir, transform=train_transform, is_train=True)
test_dataset = MyData(test_root_dir, transform=test_transform, is_train=False)

# Split training set into training and validation sets
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size

train_dataset, val_dataset = data.random_split(train_dataset, [train_size, val_size])
val_dataset.dataset.transform = test_transform

# Create data loaders
batch_size = 128

train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# Define device, model, and training hyperparameters
device = torch.device('cuda')
model = torchvision.models.resnet34(pretrained=True) # Use pre-trained ResNet34
model.fc = nn.Linear(model.fc.in_features, 10) # Since the classes are only 10, modify the final layer
model = model.to(device)
loss = nn.CrossEntropyLoss() # Define loss function
num_epochs, lr, wd = 20, 3e-4, 5e-3 # Set number of epochs, learning rate, and weight decay
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=wd) # Applying Adam
lr_period, lr_decay = 5, 0.3 # Learning rate scheduler parameters
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_period, gamma=lr_decay)
scaler = GradScaler() # Gradient scaler for mixed precision training

# Training loop
for epochs in range(num_epochs):
    model.train()
    running_loss = 0.
    total, accuracy = 0., 0. # Track training metrics
    val_loss, total_val, correct_val =0., 0., 0. # Track validation metrics

    for x_train, y_train in train_loader:
        x_train, y_train = x_train.to(device), y_train.to(device)

        with autocast(): # Use mixed precision for faster training

            out = model(x_train) # Forward pass
            l = loss(out, y_train) # Compute loss

        optimizer.zero_grad()
        scaler.scale(l).backward() # Backward pass with scaled gradients
        scaler.step(optimizer)
        scaler.update() # Update gradients and optimizer

        # Track training metrics
        running_loss += l.item() * x_train.size(0)
        _, predictor = torch.max(out.data, dim=1)
        total += y_train.size(0)
        accuracy += (predictor == y_train).sum().item()
    scheduler.step() # Update learning rate

    # Calculate average loss and accuracy
    avg_loss = running_loss / len(train_loader.dataset)
    train_accuracy = 100 * accuracy / total
    print(f'Epoch: {epochs+1}, Loss: {avg_loss:.5f}, Accuracy: {train_accuracy:.3f}%')

    # Validation loop
    model.eval()
    with torch.no_grad():  # Disable gradient calculation for validation
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            l_val = loss(outputs, labels)

            # Calculate validation loss and accuracy
            val_loss += l_val.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    # Calculate validation metrics
    avg_val_loss = val_loss / len(val_loader.dataset)
    val_accuracy = 100 * correct_val / total_val
    print(f'V_Epoch: {epochs+1}, V_Loss: {avg_val_loss:.5f}, V_Accuracy: {val_accuracy:.3f}%')

# Test phase
model.eval()
image_filename = [] # Store file names for submission
all_pred = [] # Store predictions

with torch.no_grad():
    for x_test, file_name in test_loader:
        x_test = x_test.to(device)

        with autocast():
            outputs = model(x_test)
            predictions = torch.argmax(outputs, 1).detach().cpu().numpy()
        for name in file_name:
            id, _ = os.path.splitext(name)
            image_filename.append(id)

        all_pred.extend(predictions)

# Check for any mismatch between filenames and predictions
if len(image_filename) != len(all_pred):
    raise ValueError("Mismatch between number of image filenames and number of predictions!")

# Map predictions to original labels
all_pred_label = [train_dataset.dataset.reverse_mapping[label] for label in all_pred]

# Create submission file
submission = pd.DataFrame({
    'id': image_filename ,
    'label': all_pred_label
})

submission.to_csv('submission.csv', index=False)
