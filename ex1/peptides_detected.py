import random

pos_path = "pos_A0201.txt"
neg_path = "neg_A0201.txt"
def read_file(pos_path, neg_path):
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
    print(len(pos_train_data))
    print(len(neg_train_data))
    print(len(pos_test_data))
    print(len(neg_test_data))
    return pos_train_data, neg_train_data, pos_test_data, neg_test_data

pos_train_data, neg_train_data, pos_test_data, neg_test_data = read_file(pos_path, neg_path)


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
        self.fc2 = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
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




