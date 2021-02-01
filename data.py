import torch
from torch.utils.data import DataLoader
import math
from torchvision.datasets import ImageFolder
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import config


# spcify the parameters


# create a transformer
transformer=transforms.Compose([
            transforms.Resize((config.image_size,config.image_size)),
            transforms.ToTensor()
        ])

# transform the images and apply dataloader
anime_dataset=ImageFolder('./anime_images',transformer)
dataloader = DataLoader(anime_dataset,batch_size=config.batch_size,shuffle=True)



