

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
        self.optim = torch.optim.Adam(self.parameters(), lr=0.01)
        self.criterion = nn.BCELoss()


    def forward(self, x):
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        out = torch.sigmoid(self.fc3(x))
        return out

    def train_model(self, X_train,  y_train, X_test, y_test, epoch :int, plot_data = None):
        loss = None
        for i in range(epoch):
            y_pred = self.forward(X_train)
            loss = self.criterion(y_pred, y_train)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            if plot_data:
                plot_data[0].append(loss.detach().cpu().numpy())
                plot_data[1].append(self.test_model(X_test, y_test).detach().cpu().numpy())
        return loss

    def test_model(self, test_data, y_test):
        with torch.no_grad():
            y_pred = self.forward(test_data)
            return self.criterion(y_pred, y_test)

