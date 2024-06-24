import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# import matplotlib as plt
import matplotlib.pyplot as plt
from tqdm import tqdm
from Autoencoder import Encoder

class Classifier(nn.Module):
    def __init__(self, latent_dim):
        super(Classifier, self).__init__()
        self.seq = nn.Sequential(nn.Linear(12, 128), nn.ReLU(), nn.Linear(128, 10), nn.Sigmoid())
        self.encoder = Encoder()

    def forward(self, x):
        x = self.encoder(x)
        return self.seq(x)