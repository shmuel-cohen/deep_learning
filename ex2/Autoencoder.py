import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# import matplotlib as plt
import matplotlib.pyplot as plt
from tqdm import tqdm

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

        self.seq = nn.Sequential(nn.Conv2d(1, 4, kernel_size=5, padding=2), nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                    nn.Conv2d(4, 16, kernel_size=3, padding=1),nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                  nn.Conv2d(16, 32, kernel_size=3, padding=1),nn.ReLU(),
                                  nn.Flatten(),nn.Linear(32 * 49, 128),nn.ReLU(),nn.Linear(128, 12),nn.ReLU()
                                  )

    def forward(self, x):
        return self.seq(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.seq1 = nn.Sequential(nn.Linear(12, 128), nn.ReLU(), nn.Linear(128, 32 * 49), nn.ReLU())
        self.seq2 = nn.Sequential(nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(), nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1), nn.Sigmoid())

    def forward(self, x):
        x =self.seq1(x)
        x = x.view(-1, 32, 7, 7)
        x =self.seq2(x)
        return x

# Model for #1
class Autoencoder(nn.Module):
    def __init__(self, ):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
