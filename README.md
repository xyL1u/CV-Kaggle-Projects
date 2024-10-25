# Kaggle Projects about CV
Welcome to my Kaggle Competitions repository! This repository contains the code I used to participate in several Kaggle competitions. Each competition is implemented in Python using Pytorch.

## Table of Contents
- [Introduction](#introduction)
- [Competitions Overview](#competitions-overview)
- [Setup](#setup)
- [Project Details](#project-details)

## Introduction

This repository is a collection of deep learning projects that I have worked on while I was self-studying DL. The projects demonstrate a variety of approaches to tasks like regression, image classification and object detection.

Special thanks to the Kaggle community and the deep learning course taught by Mu Li.

## Competitions Overview

Here are the Kaggle competitions I participated in, and the corresponding coed files in this repository:

1. **[California House Price Prediction](https://www.kaggle.com/competitions/california-house-prices)** - [`ca_houseprice.py`](./ca_houseprice.py)
2. **[CIFAR-10 Image Classification](https://www.kaggle.com/competitions/cifar-10)** - [`cifar10.py`](./cifar10.py)
3. **[Leaves Classification](https://www.kaggle.com/competitions/classify-leaves)** - [`leaves_classify.py`](./leaves_classify.py)
4. **[House Price Linear Regression](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)** - [`house_prices_linear.py`](./house_prices_linear.py)
5. **[Dog Breed Classification](https://www.kaggle.com/competitions/dog-breed-identification)** - [`dogbreed.py`](./dogbreed.py)

## Setup

To run these projects, you'll need Python and the following libraries:
- 'torch'
- 'torchvision'
- 'pandas'
- 'numpy'
- 'PIL'
- 'matplotlib'

All datasets were provided by Kaggle for the respective competitions.

All paths in the code are my local absolute paths. You need download the datasets and change these paths into yours.

## Project Details

### 1. California House Price Prediction

- **Goal**: Predict housing prices in California.
- **File**:['ca_houseprice.py'](./ca_houseprice.py)
- **Approach**:
  - Fully connected layers.
  - ReLU activations.
  - Dropout for regularization.
  - Pay attention to the processing of data.
 
### 2. CIFAR-10 Image Classification

- **Goal**: Classify images into 10 different categories based on the CIFAR-10 dataset.
- **File**:[`cifar10.py`](./cifar10.py)
- **Approach**:
  - Pretrained ResNet-34 model.
  - Fine-tunes the fully connected layer.
  - Learning rate decay.

 ### 3. Leaves Classification

- **Goal**: Classify different species of leaves based on given dataset.
- **File**:[`leaves_classify.py`](./leaves_classify.py)
- **Approach**:
  - A custom ResNet-50 model.

 ### 4. House Price Linear Regression

- **Goal**: Predict housing prices.
- **File**:[`house_prices_linear.py`](./house_prices_linear.py)
- **Approach**:
  - Linear regression.
  - Define cross-validation from scratch.

 ### 5. Dog Breed Classification

- **Goal**: Classify different dog breeds based on given dataset.
- **File**:[`dogbreed.py`](./dogbreed.py)
- **Approach**:
  - Pretrained ResNet-152 model.
  - Temperature scaling is applied for model calibration.
- **Note**: Scores are terrible on competitions since the evaluation of results is using Multi Class Log Loss. So the scores are very high.
