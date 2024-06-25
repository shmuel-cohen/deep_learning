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
import Classifier
import numpy as np

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
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader, test_loader

def Q1(train_loader):
    model = Autoencoder.Autoencoder()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(NUM_EPOCHS):
        train_loss = 0
        for data_batch, label_batch in train_loader:
            optimizer.zero_grad()
            output = model(data_batch)

            img_tensor = output[0].squeeze().detach().cpu().numpy()  # Squeeze and convert to numpy
            plt.imshow(img_tensor, cmap='gray')
            plt.show()

            loss = criterion(output, data_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(train_loss)

def Q2(train_loader, test_loader):
    model = Classifier.Classifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loss, test_loss = [],[]
    train_accuracy, test_accuracy = [], []
    for epoch in range(NUM_EPOCHS):
        model.train()
        agg_train_loss = 0
        correct_train = 0
        total_train = 0

        for data_batch, label_batch in train_loader:
            data_batch, label_batch = data_batch.to(model.device), label_batch.to(model.device)

            one_hot_label_batch = one_hot_encode(label_batch)
            optimizer.zero_grad()
            output = model(data_batch)
            loss = criterion(output, one_hot_label_batch)
            loss.backward()
            optimizer.step()
            agg_train_loss += loss.item()

            _, predicted = torch.max(output, 1)
            correct_train += (predicted == label_batch).sum().item()
            total_train += label_batch.size(0)

        train_loss.append(agg_train_loss / len(train_loader))


        model.eval()
        agg_test_loss = 0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for data_batch, label_batch in test_loader:
                data_batch, label_batch = data_batch.to(model.device), label_batch.to(model.device)
                one_hot_label_batch = one_hot_encode(label_batch)
                output = model(data_batch)
                loss = criterion(output, one_hot_label_batch)
                agg_test_loss += loss.item()

                _, predicted = torch.max(output, 1)
                correct_test += (predicted == label_batch).sum().item()
                total_test += label_batch.size(0)
        test_loss.append(agg_test_loss / len(test_loader))
        test_accuracy.append(100 * correct_test / total_test)

        print(f'Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {train_loss[-1]:.4f}, Test Loss: {test_loss[-1]:.4f}')


    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Curves')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy, label='Train Accuracy')
    plt.plot(test_accuracy, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy Curves')
    plt.legend()
    plt.savefig('accuracy_plot.png')
    plt.show()





def one_hot_encode(batch_labels, num_classes=10):
    # Create a zero matrix of shape (batch_size, num_classes)
    one_hot_vectors = np.zeros((len(batch_labels), num_classes))

    # Set the value at the index corresponding to each label to 1.000
    for i, label in enumerate(batch_labels):
        one_hot_vectors[i, label] = 1

    return torch.tensor(one_hot_vectors, dtype=torch.float32)


if __name__ == '__main__':
    train_loader, test_loader = data_loader()
    # Q1(train_loader)
    Q2(train_loader, test_loader)