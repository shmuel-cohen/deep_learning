import Autoencoder
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class Classifier(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Autoencoder.Autoencoder().encoder
        self.classifier = nn.Sequential(
            nn.Linear(in_features=16, out_features=12),
            nn.ReLU(),
            nn.Linear(in_features=12, out_features=10),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten()
        x = self.classifier(x)
        return x



