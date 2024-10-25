import torch
from torch import nn
import os
import pandas as pd
from PIL import Image
from torch.utils import data
import torchvision
from torchvision import transforms
from torch.amp import autocast, GradScaler
import torch.nn.functional as F

# Custom dataset class to load images and labels
class MyDataset(data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None, is_train=True):
        super().__init__()
        self.csv_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train

        # Map labels to int
        if is_train:
            self.image_labels = list(self.csv_file.itertuples(index=False, name=None))
            self.label_mapping = {label: idx for idx, label in enumerate(self.csv_file.iloc[:, 1].unique())}

        else:
            self.test_images = os.listdir(self.root_dir)

    def __len__(self):
        if self.is_train:
            return len(self.csv_file)
        else:
            return len(self.test_images)

    def __getitem__(self, idx):
        if self.is_train:
            file_name, label = self.image_labels[idx]
            image_path = os.path.join(self.root_dir, file_name + '.jpg')
            image = Image.open(image_path).convert('RGB')
            label = self.label_mapping[label]

            if self.transform:
                image = self.transform(image)

            return image, torch.tensor(label, dtype=torch.long)

        else:
            file_name = self.test_images[idx]
            image_path = os.path.join(self.root_dir, file_name)
            image = Image.open(image_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

                return image, file_name

# Define transformations for training and test data
train_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.RandomResizedCrop(224, scale=(0.64, 1.0), ratio=(3.0/4.0, 4.0/3.0)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

# Images root directories
train_root = 'train'
test_root = 'test'
csv_file = 'labels.csv'

# Create datasets and split training data into training and validation datasets
train_dataset = MyDataset(csv_file, train_root, transform=train_transforms, is_train=True)
test_dataset = MyDataset(csv_file, test_root, transform=test_transforms, is_train=False)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size

train_dataset, val_dataset = data.random_split(train_dataset, lengths=[train_size, val_size])
val_dataset.dataset.transform = test_transforms

# Create data loaders
train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = data.DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define device, model, loss function and optimizer
device = torch.device('cuda')
model = torchvision.models.resnet152(pretrained=True)
for params in model.parameters():
    params.requires_grad = False
model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 256), nn.ReLU(),
                         nn.Linear(256, 120))
model = model.to(device)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.fc.parameters(), lr=1e-3, weight_decay=0)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
# scaler = GradScaler()
num_epoch = 20

# Training loop
for epochs in range (num_epoch):
    model.train()
    total_train, loss_train, accuracy_train = 0., 0., 0.
    total_val, loss_val, accuracy_val = 0., 0., 0.

    for x_train, y_train in train_loader:
        x_train, y_train = x_train.to(device), y_train.to(device)
        # with autocast('cuda'):
        out = model(x_train)
        l = loss(out, y_train)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        # scaler.update()

        # Track training metrics
        loss_train += l.item() * x_train.size(0)
        _, pred_train = torch.max(out.data, dim=1)
        total_train += y_train.size(0)
        accuracy_train += (pred_train == y_train).sum().item()
    # scheduler.step()
    avg_loss = loss_train / len(train_loader.dataset)
    train_acc = 100 * accuracy_train / total_train
    print(f'Epoch: {epochs+1}/{num_epoch}, Loss: {avg_loss:.5f}, Accuracy: {train_acc:.3f}%')

# Temperature scaling for model calibration
class TemperatureScaling(nn.Module):
    """
        A module to perform temperature scaling for model calibration.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1)

    def forward(self, logits):
        return logits / self.temperature

def calibrate_model(model, val_loader, device, max_iter=50):
    """
        Calibrate model using temperature scaling.

        Args:
            model: The trained model to calibrate
            val_loader: Validation data loader
            device: Device to perform computations on
            max_iters: Maximum number of iterations for calibration

        Returns:
            calibrated_model: Model with temperature scaling layer
    """
    temperature_scale = TemperatureScaling().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS(temperature_scale.parameters(), lr=0.01, max_iter=max_iter)

    logits_list = []
    labels_list = []
    # Collect all logits and labels from validation dataset
    model.eval()
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            logits = model(x_val)
            logits_list.append(logits)
            labels_list.append(y_val)
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    # Optimize temperature scaling parameter
    def eval():
        optimizer.zero_grad()
        scaled_logits = temperature_scale(logits)
        loss = criterion(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(eval)

    # Create a calibrated model that include temperature scaling
    class CalibratedModel(nn.Module):
        def __init__(self, model, temperature_scale):
            super().__init__()
            self.model = model
            self.temperature_scale = temperature_scale

        def forward(self, x):
            logits = self.model(x)
            return self.temperature_scale(logits)

    return CalibratedModel(model, temperature_scale)

# Define labels for submission file
labels = sorted(train_dataset.dataset.csv_file.iloc[:, 1].unique().tolist())

def generate_submission(model, test_loader, output_csv='submission.csv'):
    """
        Generate submission file with calibrated predictions.

        Args:
            model: Calibrated model
            test_loader: Test data loader
            labels: List of class labels
            device: Device to perform computations on
            output_csv: Output CSV filename
        """
    model.eval()
    results = []

    with torch.no_grad():
        for x_test, file_names in test_loader:
            x_test = x_test.to(device)
            logits = model(x_test)
            probabilities = F.softmax(logits, dim=1).detach().cpu().numpy()

            for file_name, prob in zip(file_names, probabilities):
                name, _ = os.path.splitext(file_name)
                result = [name] + prob.tolist()
                results.append(result)

    df = pd.DataFrame(results, columns=['id'] + labels)
    df.to_csv(output_csv, index=False)

# Calibrate and generate submission
calibrate_model = calibrate_model(model, val_loader, device)
generate_submission(calibrate_model, test_loader)