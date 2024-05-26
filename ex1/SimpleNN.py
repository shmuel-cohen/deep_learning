

import torch
import torch.nn as nn
import torch.nn.functional as functional
from sklearn.metrics import roc_curve, auc

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

    # def test_model(self, test_data, y_test):
    #     with torch.no_grad():
    #         y_pred = self.forward(test_data)  # Get the raw model outputs (logits or probabilities)
    #
    #     # Assuming y_pred is a PyTorch tensor, convert it to a numpy array
    #     y_pred = y_pred.cpu().numpy()
    #
    #
    #     # Compute ROC curve and ROC area for the given threshold
    #     fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    #     roc_auc = auc(fpr, tpr)

        # return fpr, tpr, thresholds, roc_auc
    def test_model(self, test_data, y_test, threshold):
        with torch.no_grad():
            y_test = y_test.flatten().numpy()
            y_pred = self.forward(test_data)
            y_pred = (y_pred > threshold).int().flatten().numpy()
            fpn = y_pred[y_pred == 1 and y_test == 0].shape[0]
            fnn = y_pred[y_pred == 0 and y_test == 1].shape[0]
            acc = (y_test.shape[0] - y_pred[y_pred != y_test].shape[0]) / y_test.shape[0]
            return fpn, fnn, y_test[y_test == 1].shape[0], y_test[y_test == 0].shape[0], acc

