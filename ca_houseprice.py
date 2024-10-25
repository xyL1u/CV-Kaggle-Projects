import torch
from torch import nn
from torch.utils import data
from d2l import torch as d2l  # For plotting.Same as linear houseprice
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load training and testing data from CSV files
train_data = pd.read_csv('california_houseprice/train.csv')
test_data = pd.read_csv('california_houseprice/test.csv')

# Prepare features from training and testing data
n_train = train_data.shape[0]
train_f_process = pd.concat([train_data.iloc[:, 2], train_data.iloc[:, 5], train_data.iloc[:, 11:16]], axis=1)
test_f_process = pd.concat([test_data.iloc[:, 4], test_data.iloc[:, 10:15]], axis=1)

# Combine features for unified processing, then split afterward
all_features = pd.concat([train_f_process, test_f_process])

# Standardize numeric features and fill missing values with 0
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / x.std())
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# One-hot encode categorical features and convert to numerical format
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features = all_features * 1

# Convert features and labels to tensors for PyTorch
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(train_data.iloc[:, 2].values.reshape(-1, 1), dtype=torch.float32)
in_features = train_features.shape[1]  # Number of features after processing


# Define the neural network model with dropout layers for regularization
def get_net():
    net = nn.Sequential(nn.Linear(in_features, 256), nn.Dropout(0.5),
                        nn.ReLU(), nn.Linear(256, 64), nn.Dropout(0.5),
                        nn.ReLU(), nn.Linear(64, 1))
    return net


loss = nn.MSELoss()  # Define the loss function (Mean Squared Error for regression)
device = torch.device('cuda')  # Use GPU if available
net = get_net().to(device)


# Define data loader function for batching data
def data_loader(train_features, train_labels, batch_size, is_train=True):
    dataset = data.TensorDataset(train_features, train_labels)
    dataloader = data.DataLoader(dataset, batch_size, shuffle=is_train)
    return dataloader


# Calculate the log RMSE, clamping predictions to avoid extreme values
def log_rmse(net, features, labels):
    clamp_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clamp_preds), torch.log(labels)))
    return rmse.item()


# Train the model and compute losses for each epoch
def train(net, train_features, train_labels, test_features, test_labels,
          batch_size, num_epoch, lr, wd):
    train_loss, test_loss = [], []
    train_iter = data_loader(train_features, train_labels, batch_size, is_train=True)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, weight_decay=wd)

    for epoch in range(num_epoch):
        net.train()  # Set the network to training mode
        for features, labels in train_iter:
            optimizer.zero_grad()
            l = loss(net(features), labels)
            l.backward()
            optimizer.step()

        # Record training loss after each epoch
        train_loss.append(log_rmse(net, train_features, train_labels))

        # Record validation loss if validation data is provided
        if test_labels is not None:
            net.eval()
            test_loss.append(log_rmse(net, test_features, test_labels))
    return train_loss, test_loss


# Split data into k-folds and select each fold as validation in turn
def get_k_fold_data(k, i, features, labels):
    assert k > 1  # Ensure number of folds is at least 2
    fold_size = features.shape[0] // k
    features_train, labels_train = None, None

    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        features_part, labels_part = features[idx, :], labels[idx]

        if j == i:
            features_valid, labels_valid = features_part, labels_part
        elif features_train is None:
            features_train, labels_train = features_part, labels_part
        else:
            features_train = torch.cat([features_train, features_part], 0)
            labels_train = torch.cat([labels_train, labels_part], 0)

    return features_train, labels_train, features_valid, labels_valid


# Perform k-fold cross-validation
def k_fold(k, features_train, labels_train, batch_size, num_epoch, lr, wd):
    train_ls_sum, valid_ls_sum = 0, 0

    for i in range(k):
        features_train_fold, labels_train_fold, features_valid_fold, labels_valid_fold = \
            get_k_fold_data(k, i, features_train, labels_train)

        net = get_net()  # Initialize a new network for each fold
        train_loss, valid_loss = train(net=net, train_features=features_train_fold, train_labels=labels_train_fold,
                                       test_features=features_valid_fold, test_labels=labels_valid_fold,
                                       batch_size=batch_size, num_epoch=num_epoch, lr=lr, wd=wd)

        # Sum the final training and validation losses of each fold
        train_ls_sum += train_loss[-1]
        valid_ls_sum += valid_loss[-1]

        # Plot the training and validation losses for the first fold
        if i == 0:
            d2l.plot(list(range(1, num_epoch + 1)), [train_loss, valid_loss],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epoch],
                     legend=['train', 'valid'], yscale='log')
            plt.show()

        print(f'Fold {i + 1}, train log rmse {float(train_loss[-1]):f}, '
              f'valid log rmse {float(valid_loss[-1]):f}')

    return train_ls_sum / k, valid_ls_sum / k


# Set hyperparameters for k-fold cross-validation
k, batch_size, num_epoch, lr, wd = 5, 256, 100, 0.5, 0
train_loss, valid_loss = k_fold(k, train_features, train_labels, batch_size, num_epoch, lr, wd)

# Print average training and validation log RMSE across all folds
print(f'{k}-fold validation: training average log rmse: {float(train_loss):f}, '
      f'validation average log rmse: {float(valid_loss):f}')
