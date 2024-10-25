import torch
from torch import nn
import pandas as pd
import numpy as np
from torch.utils import data
from d2l import torch as d2l # Dive to deep learning library for visualization
import matplotlib.pyplot as plt

# Load training and testing data
train_data = pd.read_csv('california_houseprice/train.csv')
test_data = pd.read_csv('california_houseprice/test.csv')

# Select specific features for preprocessing in both training and test data
train_f_process = pd.concat([train_data.iloc[:, 2], train_data.iloc[:, 5], train_data.iloc[:, 11:16]], axis=1)
test_f_process = pd.concat([test_data.iloc[:, 4], test_data.iloc[:, 10:15]], axis=1) # 对训练集和测试集一起做数据处理然后再分开，更简洁一点
all_features = pd.concat([train_f_process, test_f_process])

# Identify numerica features and normalize them
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std())) # Normalization
all_features[numeric_features] = all_features[numeric_features].fillna(0) # Fill missing values with 0
all_features = pd.get_dummies(all_features, dummy_na=True) # One-hot encode categorical features
all_features = all_features * 1 # Turn True and False into 1 and 0

# Split processed features back ino training and testing datasets
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32) # [:n_train] 是切掉后面test data row(1460,81)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32) # (1459,330)
train_labels = torch.tensor(train_data.iloc[:, 2].values.reshape(-1, 1),
                                dtype=torch.float32)

# Define NN model
def get_net():
    net = nn.Sequential(nn.Linear(in_features, 256), nn.Dropout(0.5),
                        nn.ReLU(), nn.Linear(256, 64), nn.Dropout(0.5),
                        nn.ReLU(), nn.Linear(64, 1))
    return net

loss = nn.MSELoss() # Define loss function
in_features = train_features.shape[1]
device = torch.device('cuda')
net = get_net().to(device)

# Define RMSE calculation with logarithmic transformation
def log_rmse(net, features, labels):
    clipped_preds = torch.clamp(net(features), 1, float('inf')) # Limit predictions between 1 and infinity
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()

# Define data loader
def train_loader(train_features, train_labels, batch_size, is_train = True):
    dataset = data.TensorDataset(train_features, train_labels)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

# Define training function
def train(net, train_features, train_labels, test_features, test_labels,
         batch_size, num_epochs, lr, wd):
    train_loss, test_loss = [], []
    train_iter = train_loader(train_features, train_labels, batch_size, is_train=True)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, weight_decay=wd)

    for epoch in range(num_epochs):
        net.train()
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        # Log training loss at the end of each epoch
        train_loss.append(log_rmse(net, train_features, train_labels))

        # Validation and record test loss
        if test_labels is not None:
            net.eval()
            test_loss.append(log_rmse(net, test_features, test_labels))

    return train_loss, test_loss

# Split data into k-folds for cross-validation
def get_k_fold_data(k, i, X, y):
    assert k > 1 # Ensure number of folds is at least 2
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part # Validation fold
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train,y_part], 0)
    return X_train, y_train, X_valid, y_valid

# k-Fold cross-validation training
def k_fold(k, X_train, y_train, batch_size, num_epochs, lr, wd):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        X_train_fold, y_train_fold, X_valid_fold, y_valid_fold = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_loss, valid_loss = train(net, X_train_fold, y_train_fold, X_valid_fold, y_valid_fold,
                                       batch_size, num_epochs, lr, wd)
        train_l_sum += train_loss[-1]
        valid_l_sum += valid_loss[-1]

        # Plot training and validation loss
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_loss, valid_loss],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
            plt.show()

        print(f'fold{i + 1}，train log rmse{float(train_loss[-1]):f}, '
              f'valid log rmse{float(valid_loss[-1]):f}')

    return train_l_sum / k, valid_l_sum / k

# Set hyperparameters for k-fold cross-validation
k, batch_size, num_epochs, lr, wd = 5, 256, 200, 0.001, 2
train_l, valid_l = k_fold(k, train_features, train_labels,batch_size,
                          num_epochs, lr, wd)

# Print final training and validation results after k-fold cross-validation
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')

# Final training and prediction function
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_loss, _ =train(net, train_features, train_labels, None, None,
                         batch_size, num_epochs, lr, weight_decay)

    # Plot training loss over epochs
    d2l.plot(np.arange(1, num_epochs + 1), [train_loss], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    plt.show()

    print(f'训练log rmse：{float(train_loss[-1]):f}')

    # Generate predictions for test set
    preds = net(test_features).detach().numpy()
    test_data['Sold Price'] = pd.Series(preds.reshape(1, -1)[0])

    # Save predictions to CSV file for submission
    submission = pd.concat([test_data['Id'], test_data['Sold Price']], axis=1)
    submission.to_csv('submission.csv', index=False)

# Train and predict
train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, wd, batch_size)