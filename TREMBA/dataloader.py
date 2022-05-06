import time
import random 
import numpy as np
import torch 
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import math
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import random_split
import os
from PIL import Image
import torchvision.models as models
import torch.nn.functional as F


def imagenet(state):
        
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor()])

    train_loader = torch.utils.data.DataLoader(
        dset.ImageFolder(state['data_train_path'], transform=transform),
        batch_size=state['batch_size'], shuffle=True,
        num_workers=0, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        dset.ImageFolder(state['data_test_path'], transform=transform),
        batch_size=state['batch_size'], shuffle=True,
        num_workers=0, pin_memory=True)

    nlabels = 1000

    return train_loader, test_loader, nlabels, mean, std