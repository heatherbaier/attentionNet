from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import pandas as pd
import numpy as np
import torch
import os


weights = {'1': 1/2236, '2': 1/1897, '3': 1/1376, '4': 1/366}


def classify_schools(x):
    if x < 90.61:
        return weights['4']
    elif (x >= 90.61) & (x < 125.31):
        return weights['3']
    elif (x >= 125.31) & (x < 160.01):
        return weights['2']
    else:
        return weights['1']


class SchoolDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.schools_df = pd.read_csv(csv_file)
        self.schools_df['weight'] = self.schools_df['overall_mean'].apply(lambda x: classify_schools(x))
        self.image_paths = []
        self.labels = []
        self.data = []

        for image in os.listdir(root_dir):
            cur_schoolid = image.split(".")[0]
            cur_df = self.schools_df[self.schools_df['school_id'] == int(cur_schoolid)]
            self.labels.append(cur_df.overall_mean.values[0])
            self.image_paths.append(os.path.join(root_dir, image))
            self.data.append((self.loadImage(os.path.join(root_dir, image)), cur_df.overall_mean.values[0], cur_df.weight.values[0]))

    def loadImage(self, impath):
        to_tens = transforms.ToTensor()
        return to_tens(Image.open(impath).convert('RGB'))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        return self.loadImage(path), self.labels[index]



