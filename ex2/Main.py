import torch
import random
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
# import matplotlib as plt
import matplotlib.pyplot as plt
from tqdm import tqdm
import Autoencoder
import Classifier
import Pretrained_encoder
import numpy as np

NUM_EPOCHS = 10
BATCH_SIZE = 256
PATH_TO_SAVED_AE = ""
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
    return train_loader, test_loader, test_dataset

def Q1_S(train_loader, test_loader):
    model = Autoencoder.Autoencoder()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loss, test_loss = [],[]
    for epoch in range(NUM_EPOCHS):
        model.train() #why do you take it out from the next loop? we dont want train on every batch?
        agg_train_loss = 0
        correct_train = 0
        total_train = 0

        for data_batch, _ in train_loader:
            # data_batch, label_batch = data_batch.to(model.device), label_batch.to(model.device)

            # one_hot_label_batch = one_hot_encode(label_batch)
            optimizer.zero_grad()
            output = model(data_batch)
            loss = criterion(output, data_batch)
            loss.backward()
            optimizer.step()
            agg_train_loss += loss.item()

            _, predicted = torch.max(output, 1)
            correct_train += (predicted == data_batch).sum().item()
            total_train += data_batch.size(0)

        train_loss.append(agg_train_loss / len(train_loader))
        # train_accuracy.append(100 * correct_train / total_train)


        model.eval()
        agg_test_loss = 0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for data_batch, label_batch in test_loader:
                # data_batch, label_batch = data_batch.to(model.device), label_batch.to(model.device)
                # one_hot_label_batch = one_hot_encode(label_batch)
                output = model(data_batch)
                loss = criterion(output, data_batch)
                agg_test_loss += loss.item()

                _, predicted = torch.max(output, 1)
                correct_test += (predicted == data_batch).sum().item()
                total_test += data_batch.size(0)
        test_loss.append(agg_test_loss / len(test_loader))

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

    # plt.subplot(1, 2, 2)
    # # plt.plot(train_accuracy, label='Train Accuracy')
    # plt.plot(test_accuracy, label='Test Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy (%)')
    # plt.title('Training and Test Accuracy Curves')
    # plt.legend()
    # plt.savefig('accuracy_plot.png')
    # plt.show()
    # print(train_accuracy)
    return model

def Q1(train_loader):
    model = Autoencoder.Autoencoder()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

    return model

def Q2(train_loader, test_loader, trained_encoder= None):
    model = Classifier.Classifier()
    if trained_encoder is not None:
        model.encoder = trained_encoder
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loss, test_loss = [],[]
    test_accuracy = []
    for epoch in range(NUM_EPOCHS):
        model.train() #why do you take it out from the next loop? we dont want train on every batch?
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
    plt.plot(test_accuracy, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy Curves')
    plt.legend()
    plt.savefig('accuracy_plot.png')
    plt.show()
    return model.encoder


def Q3(train_loader, test_data, pretrained_encoder, AE,):
    pre_AE = train_AE_with_fixed_encoder(pretrained_encoder, train_loader)
    compare_RC_images(AE, pre_AE, test_data, 10)


def train_AE_with_fixed_encoder(pretrained_encoder, train_loader):
    model = Pretrained_encoder.PretrainedEncoderAE(pretrained_encoder)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    for epoch in range(NUM_EPOCHS):
        train_loss = 0
        for data_batch, label_batch in train_loader:
            optimizer.zero_grad()
            output = model(data_batch)
            loss = criterion(output, data_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

    return model


def compare_RC_images(AE, pre_AE, test_dataset, num_images):
    original_images = []
    AE_RC = []
    preAE_RC =[]

    random_indices = random.choices(range( len(test_dataset)), k=num_images)
    subset = Subset(test_dataset, random_indices)

    for img, label in subset:
        img = img.unsqueeze(0)

        AE_out= AE(img)
        preAE_out = pre_AE(img)

        original_images.append(img)
        AE_RC.append(AE_out)
        preAE_RC.append(preAE_out)

    fig, axes = plt.subplots(3, num_images, figsize=(15, 5))

    # Plot original images in the first row
    for i in range(num_images):
        original_img = original_images[i].squeeze(0).squeeze(0)
        axes[0, i].imshow(original_img.detach().numpy())
        axes[0, i].axis('off')  # Hide axes

    # Plot reconstructed images in the second row
    for i in range(num_images):
        AE_RC_image = AE_RC[i].squeeze(0).squeeze(0)
        axes[1, i].imshow(AE_RC_image.detach().numpy())
        axes[1, i].axis('off')  # Hide axes

    for i in range(num_images):
        preAE_RC_image = preAE_RC[i].squeeze(0).squeeze(0)
        axes[2, i].imshow(preAE_RC_image.detach().numpy())
        axes[2, i].axis('off')  # Hide axes

    # Set the title for each row
    axes[0, 0].set_title('Original Images', fontsize=16, loc='left')
    axes[1, 0].set_title('AE reconstructed images', fontsize=16, loc='left')
    axes[2, 0].set_title('Pretrained encoder AE reconstructed Images', fontsize=16, loc='left')

    plt.tight_layout()
    plt.show()






def one_hot_encode(batch_labels, num_classes=10):
    # Create a zero matrix of shape (batch_size, num_classes)
    one_hot_vectors = np.zeros((len(batch_labels), num_classes))

    # Set the value at the index corresponding to each label to 1
    for i, label in enumerate(batch_labels):
        one_hot_vectors[i, label] = 1

    return torch.tensor(one_hot_vectors, dtype=torch.float32)

def Q4(test_loader, trained_encoder= None):
    indices = torch.arange(100)
    # BATCH_SIZE = 256
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MNIST(root='./data', train=True,
                                   transform=transform, download=True)

    train_loader_CLS = Subset(train_dataset, indices)

    train_loader = DataLoader(train_loader_CLS, batch_size=BATCH_SIZE,
                              shuffle=True)


    # train_loader_CLS = torch.utils.data.DataLoader(train_loader_CLS,
    #                                                batch_size=BATCH_SIZE,
    #                                                shuffle=True, num_workers=0)
    Q2(train_loader, test_loader, trained_encoder)

def Q5(test_loader, encoder):
    Q4(test_loader, encoder)


if __name__ == '__main__':
    train_loader, test_loader, test_data = data_loader()
    AE = Q1_S(train_loader, test_loader)
    # Q2(train_loader, test_loader)
    # AE= Q1(train_loader)
    # pretrained_encoder = Q2(train_loader, test_loader)
    #
    # AE = Autoencoder.Autoencoder()
    # pretrained_encoder = Autoencoder.Encoder()

    # Q3(train_loader, test_data, pretrained_encoder, AE)

    # Q4(test_loader)
    Q5(test_loader, AE.encoder)