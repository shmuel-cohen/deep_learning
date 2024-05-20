import random
from typing import Tuple, Any
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import SimpleNN as snn

AMINO_ACID_NUMBER = 20

AMINO_ACID_DICT = {
    "A": 0, "C": 1, "D": 2, "E": 3, "F": 4,
    "G": 5, "H": 6, "I": 7, "K": 8, "L": 9,
    "M": 10, "N": 11, "P": 12, "Q": 13, "R": 14,
    "S": 15, "T": 16, "V": 17, "W": 18, "Y": 19
}


class k_mer():
    def __init__(self, k_size: int, seq: str, label: bool):
        self.k = k_size
        self.seq = seq
        self.vector = self.convert_to_binary_vector()
        self.label = label

    def convert_to_binary_vector(self):
        one_hot_code = np.zeros(shape=self.k*AMINO_ACID_NUMBER)
        for i in range(self.k):
            one_hot_code[i*AMINO_ACID_NUMBER + AMINO_ACID_DICT[self.seq[i]]] =1
        return one_hot_code

class peptide_detected():
    def __init__(self, path_to_pos_sample, path_to_neg_sample):
        self.simple_model = None
        self.advance_model = None
        self.loss_date = [[], []]
        self.X_train, self.y_train, self.X_test, self.y_test =\
            self.process_data(path_to_pos_sample,path_to_neg_sample)

    def read_file(self, pos_path: str, neg_path: str) -> tuple[Any, Any, Any, Any]:
        def read_data_from_file(file_name, label, length=9):
            try:
                with open(file_name, "r") as file:
                    lines = file.readlines()
                    lines = [line.strip() for line in lines if line.strip()]
                    k_mers = []
                    for line in lines:
                        k_mers.append(k_mer(length, line, label))
                    return k_mers

            except Exception as e:
                print("can't open file. An error occurred:", e)
                raise Exception(e)

        def make_matrix(data):
            return pd.DataFrame([peptide.vector for peptide in data]).values, pd.DataFrame([peptide.label for peptide in data]).values

        # Call the function to read data from file
        pos_peptides = read_data_from_file(pos_path, 1)
        neg_peptides = read_data_from_file(neg_path, 0)

        # Take 90% of the data randomly
        train_data = random.sample(pos_peptides, k=int(len(pos_peptides) * 0.9)) +\
                     random.sample(neg_peptides, k=int(len(neg_peptides) * 0.9))
        # np.random.shuffle(train_data) #todo why needed

        # Get the remaining 10% of the data
        test_data = [peptide for peptide in pos_peptides if peptide not in train_data] \
                    + [peptide for peptide in neg_peptides if peptide not in train_data]
        # np.random.shuffle(test_data) #todo why needed
        train_data_X, train_data_y = make_matrix(train_data)
        test_data_X,  test_data_y = make_matrix(test_data)
        return train_data_X, train_data_y, test_data_X,  test_data_y



# ### from gpt
# import torch.nn as nn
# import torch.nn.functional as functional
# import torch.optim as optim
#
# # Define a simple neural network
# class SimpleNN(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, 1)
#
#         self.optim = torch.optim.Adam(self.parameters(), lr=0.01)
#         self.criterion = nn.BCELoss()
#
#
#     def forward(self, x):
#         x = functional.relu(self.fc1(x))
#         x = functional.relu(self.fc2(x))
#         out = torch.sigmoid(self.fc3(x))
#         return out
#
#     def train_model(self, data, y_true, epoch :int, plot_data = None):
#         loss = None
#         for i in range(epoch):
#             y_pred = self.forward(data)
#             loss = self.criterion(y_pred, y_true)
#             # if i % 10 == 0:
#             #     print(f"epoch {i}: Loss: {loss}")
#             self.optim.zero_grad()
#             loss.backward()
#             self.optim.step()
#             if plot_data:
#                 plot_data[0].append(loss.detach().cpu().numpy())
#                 plot_data[1].append(self.test_model(X_test, y_test).detach().cpu().numpy())
#         return loss
#
#     def test_model(self, test_data, y_test):
#         with torch.no_grad():
#             y_pred = self.forward(test_data)
#             return self.criterion(y_pred, y_test)


    def process_data(self, pos_path, neg_path ):
        train_data_X, train_data_y, test_data_X, test_data_y = self.read_file(pos_path, neg_path)
        X_train = torch.FloatTensor(train_data_X)
        X_test = torch.FloatTensor(test_data_X)
        y_train = torch.FloatTensor(train_data_y)  # Assuming y_train is binary (0 or 1)
        y_test = torch.FloatTensor(test_data_y)  # Assuming y_test is binary (0 or 1)
        return X_train, y_train, X_test, y_test

    def create_model(self, model:str ):
        if model == "SimpleNN":
            self.simple_model = snn.SimpleNN(180, 180)

    def train_model(self, model:str, epoch:int):
        if model == "SimpleNN":
            self.simple_model.train_model(self.X_train, self.y_train,
                                          self.X_test, self.y_test,
                                          epoch, self.loss_date)

    def plot_loss(self):
        train_loss = self.loss_date[0]
        test_loss = self.loss_date[1]

        # Plotting the training and test losses
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(train_loss)), train_loss, label='Training Loss')
        plt.plot(range(len(test_loss)), test_loss, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    pos_path = "pos_A0201.txt"
    neg_path = "neg_A0201.txt"
    p_detected = peptide_detected(pos_path, neg_path)
    p_detected.create_model("SimpleNN")
    p_detected.train_model("SimpleNN", 100)
    p_detected.plot_loss()
    # epochs = range(1, 100, 10)  # Epochs from 0 to 20 with a step of 10
    # epochs = 100
    # model = SimpleNN(180, 180)
    # plot_data =  [[],[]]
    # model.train_model(X_train, y_train, epochs, plot_data)
    # # Convert lists to numpy arrays for plotting
    # train_loss = plot_data[0]
    # test_loss = plot_data[1]
    #
    # # Plotting the training and test losses
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(epochs), train_loss, label='Training Loss')
    # plt.plot(range(epochs), test_loss, label='Test Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training and Test Loss over Epochs')
    # plt.legend()
    # plt.grid(True)
    # plt.show()





