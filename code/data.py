import torch
from torch.utils.data import DataLoader
import math
from torchvision.datasets import ImageFolder
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


def show_sample_image(dataloader):
  iterator = iter(dataloader)
  sample_batch, _ = iterator.next()
  first_sample_image_of_batch = sample_batch[0]
  print(first_sample_image_of_batch.size())
  print("Current range: {} to {}".format(first_sample_image_of_batch.min(), first_sample_image_of_batch.max()))
  plt.imshow(np.transpose(first_sample_image_of_batch.numpy(), (1, 2, 0)))

batch_size = 128
image_size = 28
transformer=transforms.Compose([
            transforms.Resize((28,28)),
            transforms.ToTensor()
        ])
anime_dataset=ImageFolder('./anime_images',transformer)
dataloader = DataLoader(anime_dataset,batch_size=batch_size,shuffle=True)
show_sample_image(dataloader)



