import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# import matplotlib as plt
import matplotlib.pyplot as plt
from tqdm import tqdm
import Autoencoder

NUM_EPOCHS = 15
BATCH_SIZE = 256
def data_loader():
    global train_loader, test_loader
    BATCH_SIZE = 256
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader

def Q1(train_loader):
    learning_rate = 0.001
    model = Autoencoder.Autoencoder()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(NUM_EPOCHS):
        train_loss = 0
        for data_batch, label_batch in train_loader:
            optimizer.zero_grad()
            output = model(data_batch)
            loss = criterion(output, data_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(train_loss)

if __name__ == '__main__':
    train_loader, test_loader = data_loader()
    Q1(train_loader)