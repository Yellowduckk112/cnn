import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils import data
from torchvision import transforms
import torchvision

transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])