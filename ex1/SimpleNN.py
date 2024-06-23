

import torch
import torch.nn as nn
import torch.nn.functional as functional
from sklearn import metrics

from sklearn.metrics import roc_curve, auc

import torch.optim as optim

# Define a simple neural network
class SimpleNN(nn.Module):
    """
    Defines a simple neural network model for binary classification.

    Parameters:
        input_size (int): The number of input features.
        hidden_size_1 (int): The number of neurons in the first hidden layer.
        hidden_size_2 (int): The number of neurons in the second hidden layer.
        weight_fn (float): Weight for false negatives.
        weight_fp (float): Weight for false positives.
        linear (bool, optional): Flag to indicate whether to use a linear activation function (default is False).
    """
    def __init__(self, input_size, hidden_size_1, hidden_size_2, weight_fn, weight_fp, linear= False):
        super(SimpleNN, self).__init__()
        self.linear= linear
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, 1)
        self.optim = torch.optim.Adam(self.parameters(), lr=0.01)
        # self.criterion = nn.BCELoss(weight=weight)
        pos_weight = torch.tensor([weight_fp/weight_fn])
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)



    def forward(self, x):
        """
        Defines the forward pass of the neural network.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # return self.fc3(self.fc2(self.fc1(x)))
        if self.linear:
            f = lambda x: x
        else:
            f = functional.relu
        x = f(self.fc1(x))
        x = f(self.fc2(x))
        out = torch.tanh_(self.fc3(x)) #- 0.5
        return out

    def train_model(self, X_train,  y_train, X_test, y_test, epoch :int, plot_data = None):
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
        for i in range(epoch):
            y_pred = self.forward(X_train)
            loss = self.criterion(y_pred, y_train)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            if plot_data:
                plot_data[0].append(loss.detach().cpu().numpy())
                plot_data[1].append(self.find_loss(X_test, y_test).detach().cpu().numpy())
        return loss

    def find_loss(self, test_data, y_test):
        """
          Computes the loss on test data.

          Parameters:
              test_data (torch.Tensor): Input test data.
              y_test (torch.Tensor): Target test data.

          Returns:
              torch.Tensor: Loss value.
          """
        with torch.no_grad():
            y_pred = self.forward(test_data)
            return self.criterion(y_pred, y_test)

    def roc_carve(self, test_data, y_test):
        """
           Computes ROC curve and AUC.

           Parameters:
               test_data (torch.Tensor): Input test data.
               y_test (torch.Tensor): Target test data.

           Returns:
               tuple: Tuple containing FPR, TPR, thresholds, and ROC AUC.
           """
        with torch.no_grad():
            y_pred = self.forward(test_data)  # Get the raw model outputs (logits or probabilities)

        # Assuming y_pred is a PyTorch tensor, convert it to a numpy array
        y_pred = y_pred.cpu().numpy()
        # Compute ROC curve and ROC area for the given threshold
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)

        return fpr, tpr, thresholds, roc_auc
    def test_model(self, test_data, y_test, threshold):
        """
               Tests the model on test data using a specified threshold.

               Parameters:
                   test_data (torch.Tensor): Input test data.
                   y_test (torch.Tensor): Target test data.
                   threshold (float): Threshold value for classification.

               Returns:
                   tuple: Tuple containing true negatives, false positives, false negatives, true positives, and accuracy.
               """
        with torch.no_grad():
            y_test = y_test.flatten()
            y_pred = self.forward(test_data)
            y_pred = (y_pred > threshold).int()
            tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
            acc = (tp + tn) / (tn + fp + fn + tp)
            return tn, fp, fn, tp, acc

