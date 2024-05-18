import random
from typing import Tuple, Any

import numpy as np
import pandas as pd

AMINO_ACID_NUMBER = 20

AMINO_ACID_DICT = {
    "A": 0, "C": 1, "D": 2, "E": 3, "F": 4,
    "G": 5, "H": 6, "I": 7, "K": 8, "L": 9,
    "M": 10, "N": 11, "P": 12, "Q": 13, "R": 14,
    "S": 15, "T": 16, "V": 17, "W": 18, "Y": 19
}

class k_mer():
    def __init__(self, k_size: int, seq: str, label : bool):
        self.k = k_size
        self.seq = seq
        self.vector= convert_to_binary_vector(self)
        self.label= label

def read_file(pos_path: str, neg_path: str) -> tuple[Any, Any, Any, Any]:
    def read_data_from_file(file_name, label, length = 9):
        try:
            with open(file_name, "r") as file:
                lines = file.readlines()
                lines = [line.strip() for line in lines if line.strip()]
                k_mers = []
                for line in lines:
                    k_mers.append(k_mer(length, line, label))
                return k_mers

        except Exception as e:
            print("An error occurred:", e)
            return []

    def make_matrix(data):
        return pd.DataFrame([peptide.vector for peptide in data]).values, pd.DataFrame([peptide.label for peptide in data]).values

    # Call the function to read data from file
    pos_peptides = read_data_from_file(pos_path, 1)
    neg_peptides = read_data_from_file(neg_path, 0)

    # Take 90% of the data randomly
    train_data = random.sample(pos_peptides,
                                   k=int(len(pos_peptides) * 0.9)) +\
                 random.sample(neg_peptides, k=int(len(neg_peptides) * 0.9))
    np.random.shuffle(train_data)

    # Get the remaining 10% of the data
    test_data = [peptide for peptide in pos_peptides if
                     peptide not in train_data] + [peptide for peptide in neg_peptides if
                     peptide not in train_data]
    np.random.shuffle(test_data)


    train_data_X, train_data_y = make_matrix(train_data)
    test_data_X,  test_data_y = make_matrix(test_data)
    return train_data_X, train_data_y, test_data_X,  test_data_y

def convert_to_binary_vector(peptide: k_mer):
    one_hot_code = np.zeros(shape = peptide.k*AMINO_ACID_NUMBER)
    for i in range(peptide.k):
        one_hot_code[i*AMINO_ACID_NUMBER + AMINO_ACID_DICT[peptide.seq[i]]] =1
    return one_hot_code

### from gpt
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)


    def forward(self, x):
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        out = torch.sigmoid(self.fc3(x))
        return out



def train_model():
    model = SimpleNN(180, 180)
    pos_path = "pos_A0201.txt"
    neg_path = "neg_A0201.txt"
    train_data_X, train_data_y, test_data_X, test_data_y = read_file(pos_path, neg_path)
    X_train = torch.FloatTensor(train_data_X)
    X_test = torch.FloatTensor(test_data_X)
    y_train = torch.FloatTensor(train_data_y)  # Assuming y_train is binary (0 or 1)
    y_test = torch.FloatTensor(test_data_y)  # Assuming y_test is binary (0 or 1)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epoch = 100
    losses = []
    for i in range(epoch):
        y_pred = model.forward(X_train)
        loss = criterion(y_pred, y_train)
        losses.append(loss)
        if i % 10 == 0:
            print(f"epoch {i}: {loss}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


train_model()





