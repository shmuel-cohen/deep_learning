import random
import numpy as np

AMINO_ACID_NUMBER = 20


def read_file(pos_path: str, neg_path: str) -> tuple[list[str],list[str],list[str],list[str]]:
    def read_data_from_file(file_name):
        try:
            with open(file_name, "r") as file:
                lines = file.readlines()
                lines = [line.strip() for line in lines if line.strip()]
                return lines
        except Exception as e:
            print("An error occurred:", e)
            return []

    # Call the function to read data from file
    pos_peptides = read_data_from_file(pos_path)
    neg_peptides = read_data_from_file(neg_path)

    # Take 90% of the data randomly
    pos_train_data = random.sample(pos_peptides,
                                   k=int(len(pos_peptides) * 0.9))
    neg_train_data = random.sample(neg_peptides,
                                   k=int(len(neg_peptides) * 0.9))

    # Get the remaining 10% of the data
    pos_test_data = [peptide for peptide in pos_peptides if
                     peptide not in pos_train_data]
    neg_test_data = [peptide for peptide in neg_peptides if
                     peptide not in neg_train_data]

    return pos_train_data, neg_train_data, pos_test_data, neg_test_data


def convert_to_binary_vector(peptides_list: list[str], amino_acid_dict: dict):
    num_samples = len(peptides_list)
    peptide_len = len(peptides_list[0])
    input_size = len(peptides_list[0] * AMINO_ACID_NUMBER)
    data = np.zeros(num_samples, input_size)
    for i in range(num_samples):
        for j in range(peptide_len):
            data[j*AMINO_ACID_NUMBER + amino_acid_dict[peptides_list[j]], i] =1
    return data


### from gpt
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax()


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.softmax(out)
        return out

# Define training function
def train_model(model, criterion, optimizer, train_loader, test_loader,
                num_epochs):
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Test the model
        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                test_loss = criterion(outputs, targets)
                running_test_loss += test_loss.item()

        test_loss = running_test_loss / len(test_loader)
        test_losses.append(test_loss)

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    return train_losses, test_losses


def main():
    pos_path = "pos_A0201.txt"
    neg_path = "neg_A0201.txt"
    pos_train_data, neg_train_data, pos_test_data, neg_test_data = read_file(pos_path, neg_path)
    amino_acid_dict = {
        "A": 0, "C": 1, "D": 2, "E": 3, "F": 4,
        "G": 5, "H": 6, "I": 7, "K": 8, "L": 9,
        "M": 10, "N": 11, "P": 12, "Q": 13, "R": 14,
        "S": 15, "T": 16, "V": 17, "W": 18, "Y": 19
    }

    # Assuming read_file function returns numpy arrays
    pos_train_data, neg_train_data, pos_test_data, neg_test_data = read_file(
        pos_path, neg_path)

    # Concatenate positive and negative training data
    X_train = np.concatenate((convert_to_binary_vector(pos_train_data, amino_acid_dict),
                              convert_to_binary_vector(neg_train_data, amino_acid_dict)), axis=0)

    # Create labels for training data
    y_train_pos = np.ones((len(pos_train_data), 1))  # Positive class label
    y_train_neg = np.zeros((len(neg_train_data), 1))  # Negative class label
    y_train = np.concatenate((y_train_pos, y_train_neg), axis=0)



    # Generate random data
    num_samples = 1000
    input_size = 180
    output_size = 1

    X_train = np.random.randn(num_samples, input_size).astype(np.float32)
    y_train = np.random.randint(2, size=(num_samples, output_size)).astype(
        np.float32)

    X_test = np.random.randn(num_samples // 2, input_size).astype(np.float32)
    y_test = np.random.randint(2, size=(num_samples // 2, output_size)).astype(
        np.float32)

    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train)

    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test)

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor,
                                                   y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32,
                                              shuffle=False)

    # Define the model
    model = MLP(input_size, 180, output_size)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 20
    train_losses, test_losses = train_model(model, criterion, optimizer,
                                            train_loader, test_loader,
                                            num_epochs)

    # Plot the training and test losses
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Test Losses')
    plt.legend()
    plt.show()





