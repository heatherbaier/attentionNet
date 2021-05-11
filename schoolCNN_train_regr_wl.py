from __future__ import print_function, division
# from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
import torchvision.models as models
# import matplotlib.pyplot as plt
import torch.optim as optim
from PIL import Image
import numpy as np
import random
import torch
import copy
import time
import os

from image_loader import SchoolDataset
from helpers import *



dataset = SchoolDataset("./data/y1314_AllSubjects.csv", "./data/imagery/")

x_train, y_train, x_val, y_val, w_train, w_val = train_test_split_regr(dataset, .80)

train = [(k,v, w) for k,v,w in zip(x_train, y_train, w_train)]
val = [(k,v, w) for k,v,w in zip(x_val, y_val, w_val)]


print(len(train))
print(len(val))

dataset_sizes = {}
dataset_sizes['train'] = len(train)
dataset_sizes['val'] = len(val)

batchSize = 32

# Prep the training and validation set
train = torch.utils.data.DataLoader(train, batch_size = batchSize, shuffle = True)
val = torch.utils.data.DataLoader(val, batch_size = batchSize, shuffle = True)


dataloaders = {}
dataloaders['train'] = train
dataloaders['val'] = val




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = torch.nn.Linear(num_ftrs, 1)
model_ft = model_ft.to(device)
criterion = torch.nn.L1Loss(reduction = 'mean')
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


model_ft = train_regr_model(model_ft, optimizer_ft, criterion, dataloaders, dataset_sizes, device, batchSize, num_epochs = 25)


torch.save({
            'epoch': 25,
            'model_state_dict': model_ft.state_dict(),
            'optimizer_state_dict': optimizer_ft.state_dict(),
            'loss': criterion,
        }, "./trained_models/schoolCNN_L0001_r50_wl.torch")