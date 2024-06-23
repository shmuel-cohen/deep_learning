import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# Define the autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(49,12)
        )
        self.decoder = nn.Sequential(
            nn.Linear(12,49),
            nn.Unflatten(1,(1,7,7)),
            nn.ConvTranspose2d(1, 4,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 16,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.ReLU(),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.encoder(x)
        pass
        x = self.decoder(x)
        return x

def run_autoencoder():
    # Initialize the autoencoder
    model = Autoencoder()

    # Define transform
    # transform = transforms.Compose([transforms.ToTensor()])
    transform = transforms.Compose([transforms.PILToTensor()])
    # Load dataset
    # Transforms images to a PyTorch Tensor

    # Download the MNIST Dataset
    dataset = datasets.MNIST(root="./data",
                             train=True,
                             download=True,
                             transform=transform)


    # DataLoader is used to load the dataset
    # for training
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=256,)

    # Define the loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the autoencoder
    num_epochs = 30
    toprint = True
    for epoch in range(num_epochs):
        for data in train_loader:
            optimizer.zero_grad()
            img, _ = data
            img = img/255
            output = model(img)
            loss = criterion(output, img)
            loss.backward()
            optimizer.step()
        # if epoch % 5 == 0:
        print('Epoch [{}/{}], Loss: {:.8f}'.format(epoch + 1, num_epochs, loss.item()))

    # Save the model
    torch.save(model.state_dict(), 'conv_autoencoder.pth')


if __name__ == '__main__':
    run_autoencoder()