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

class PretrainedEncoderAE(nn.Module):
    def __init__(self, encoder: Encoder):
        super(PretrainedEncoderAE, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        # Freeze all the parameters
        self.encoder = encoder
        for param in encoder.parameters():
            param.requires_grad = False

        self.seq1 = nn.Sequential(nn.Linear(12, 128), nn.ReLU(), nn.Linear(128, 32 * 49), nn.ReLU())
        self.seq2 = nn.Sequential(nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
                                  nn.ReLU(),
                                  nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
                                  nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.seq1(x)
        x = x.view(-1, 32, 7, 7)
        x = self.seq2(x)
        return x


