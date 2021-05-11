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


def weighted_loss(pred, true, weight):
    return torch.sum((abs(pred - true) * weight))


def train_test_split(dataset, split):
    train_num = int(len(dataset.data) * split)
    val_num = int(len(dataset.data) - train_num)

    train_indices = random.sample(range(len(dataset.data)), train_num)
    val_indices = [i for i in range(len(dataset.data)) if i not in train_indices]

    x_train = [dataset.data[i][0] for i in train_indices]   
    x_val = [dataset.data[i][0] for i in val_indices]
    y_train = [torch.tensor(dataset.data[i][1], dtype = torch.long) for i in train_indices]
    y_val = [torch.tensor(dataset.data[i][1], dtype = torch.long) for i in val_indices]

    return x_train, y_train, x_val, y_val



def train_test_split_regr(dataset, split):
    train_num = int(len(dataset.data) * split)
    val_num = int(len(dataset.data) - train_num)

    train_indices = random.sample(range(len(dataset.data)), train_num)
    val_indices = [i for i in range(len(dataset.data)) if i not in train_indices]

    x_train = [dataset.data[i][0] for i in train_indices]   
    x_val = [dataset.data[i][0] for i in val_indices]
    y_train = [torch.tensor(dataset.data[i][1], dtype = torch.float32) for i in train_indices]
    y_val = [torch.tensor(dataset.data[i][1], dtype = torch.float32) for i in val_indices]
    w_train = [torch.tensor(dataset.data[i][2], dtype = torch.float32) for i in train_indices]
    w_val = [torch.tensor(dataset.data[i][2], dtype = torch.float32) for i in val_indices]


    return x_train, y_train, x_val, y_val, w_train, w_val



def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25):

    epoch_num = 0

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, weights in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            # if phase == 'train':
            #     scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                # # Save each epoch that achieves a higher accuracy than the current best_acc in case the model crashes mid-training
                # model_name = './Philippines/AllSubjects/E2_Static/epochs/StaticResNet_Epoch' + str(epoch_num) + '.sav'
                # pickle.dump(model, open(model_name, 'wb'))

        epoch_num += 1

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



def mae(real, pred):
    '''
    Calculates MAE of an epoch
    '''
    return torch.abs(real - pred).mean()


def train_regr_model(model, optimizer, criterion, dataloaders, dataset_sizes, device, batch_size, num_epochs=25):

    epoch_num = 0

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_mae = 9000000000000000000

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # Set model to training mode
                model.train() 
            else:
              # Set model to evaluate mode
                model.eval()  

            running_loss = 0.0
            running_mae = 0

            # Iterate over data.
            for inputs, labels, weights in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)
                weights = weights.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward - track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    # loss = criterion(outputs, labels.view(-1, 1))
                    loss = weighted_loss(outputs, labels.view(-1, 1), weights.view(-1, 1))

                    # print(labels.view(-1, 1))
                    # print(outputs)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_mae += mae(outputs, labels.view(-1, 1)).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_mae = running_mae / dataset_sizes[phase]

            print('{} Loss: {:.4f} MAE: {:.4f}'.format(
                phase, epoch_loss, epoch_mae))

            # deep copy the model
            if phase == 'val' and epoch_mae < best_mae:
                best_mae = epoch_mae
                best_model_wts = copy.deepcopy(model.state_dict())
                print("Updating model weights.")

        epoch_num += 1


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_mae))
    print("\n")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model