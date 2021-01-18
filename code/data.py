import torch
from torch.utils.data import DataLoader
import math
from torchvision.datasets import ImageFolder
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


# spcify the parameters
batch_size = 128
image_size = 28

# create a transformer
transformer=transforms.Compose([
            transforms.Resize((28,28)),
            transforms.ToTensor()
        ])

# transform the images and apply dataloader
anime_dataset=ImageFolder('/Users/ipsihou/Documents/anime/anime_images',transformer)
dataloader = DataLoader(anime_dataset,batch_size=batch_size,shuffle=True)



