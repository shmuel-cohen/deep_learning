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


def convert_to_binary_vector(peptides_list, amino_acid_dict):
    def peptide_to_binary_vector(peptide):
        n = len(peptide)
        vec = np.zeros(n * AMINO_ACID_NUMBER)
        for i in range(n):
            vec[i * AMINO_ACID_NUMBER + amino_acid_dict[peptide[i]]] = 1
        return vec
    return [peptide_to_binary_vector(peptide) for peptide in peptides_list] # todo maybe return as matrix instead of list


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
def train(model, train_data, optimizer, criterion, num_epochs=100):
    for epoch in range(num_epochs):
        total_loss = 0
        for data, target in train_data:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, total_loss))

# Prepare data for training
pos_train_data_tensor = torch.tensor(pos_train_data, dtype=torch.float32)
neg_train_data_tensor = torch.tensor(neg_train_data, dtype=torch.float32)
pos_train_labels_tensor = torch.ones(pos_train_data_tensor.size(0), 1)
neg_train_labels_tensor = torch.zeros(neg_train_data_tensor.size(0), 1)

# Concatenate positive and negative training data
train_data = torch.cat((pos_train_data_tensor, neg_train_data_tensor), 0)
train_labels = torch.cat((pos_train_labels_tensor, neg_train_labels_tensor), 0)

# Combine data and labels
train_data = torch.cat((train_data, train_labels), 1)

# Shuffle the data
train_data = train_data[torch.randperm(train_data.size(0))]

# Define the model, loss function, and optimizer
input_size = len(pos_train_data[0])
hidden_size = 10
model = SimpleNN(input_size, hidden_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train(model, train_data, optimizer, criterion, num_epochs=100)



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
    print(convert_to_binary_vector(pos_train_data, amino_acid_dict)[0], "\n",
          pos_train_data[0])



