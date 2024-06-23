import Autoencoder
import torch
import torch.nn as nn
import torch.optim as op
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.encoder = Autoencoder.Autoencoder().encoder
        self.classifier = nn.Sequential(
            nn.Linear(in_features=9, out_features=30),
            nn.ReLU(),
            nn.Linear(in_features=30, out_features=10)
        )
        self.optim = op.Adam(self.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_model(self, train_loader, num_epoch :int):
        """
        Trains the model.

        Parameters:
            X_train (torch.Tensor): Input training data.
            y_train (torch.Tensor): Target training data.
            X_test (torch.Tensor): Input test data.
            y_test (torch.Tensor): Target test data.
            epoch (int): Number of epochs to train.
            plot_data (list, optional): Data for plotting loss.

        Returns:
            torch.Tensor: Loss value.
        """
        loss = None
        for epoch in range(num_epoch):
            for data in train_loader:
                self.optim.zero_grad()
                img, label = data
                img = img / 255
                output = model(img)
                loss = self.criterion(output, label)
                loss.backward()
                self.optim.step()
            # if epoch % 5 == 0:
            print('Epoch [{}/{}], Loss: {:.8f}'.format(epoch + 1, epoch, loss.item()))

if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
    # transform = transforms.Compose([transforms.Pad(2), transforms.PILToTensor()])
    data= datasets.MNIST(root="./data",
                             train=True,
                             download=True,
                             transform=transform)


    # DataLoader is used to load the dataset
    # for training
    train_loader = torch.utils.data.DataLoader(dataset=data,
                                         batch_size=32, shuffle = True)

    model = Classifier()
    model.train_model(train_loader, 5)



