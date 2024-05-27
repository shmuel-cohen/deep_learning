import random
from typing import Tuple, Any
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import SimpleNN as snn

AMINO_ACID_NUMBER = 20

POS_PATH = "pos_A0201.txt"
NEG_PATH = "neg_A0201.txt"
AMINO_ACID_DICT = {
    "A": 0, "C": 1, "D": 2, "E": 3, "F": 4,
    "G": 5, "H": 6, "I": 7, "K": 8, "L": 9,
    "M": 10, "N": 11, "P": 12, "Q": 13, "R": 14,
    "S": 15, "T": 16, "V": 17, "W": 18, "Y": 19
}


class K_mer():
    """
    Class representing a k-mer sequence.

    Attributes:
        k (int): Length of the k-mer.
        seq (str): Sequence of amino acids.
        vector (np.array): Binary vector representation of the k-mer.
        label (bool): Label indicating positive (True) or negative (False).
    """

    def __init__(self, k_size: int, seq: str, label: int =None):
        """
        Initialize a k-mer instance.

        Args:
            k_size (int): Length of the k-mer.
            seq (str): Sequence of amino acids.
            label (bool): Label indicating positive (True) or negative (False).
        """
        self.k = k_size
        self.seq = seq
        self.vector = self.convert_to_binary_vector()
        self.label = label

    def convert_to_binary_vector(self):
        """
        Convert the k-mer sequence to a binary vector representation.

        Returns:
            np.array: Binary vector representation of the k-mer.
        """
        one_hot_code = np.zeros(shape=self.k * AMINO_ACID_NUMBER)
        for i in range(self.k):
            one_hot_code[i * AMINO_ACID_NUMBER + AMINO_ACID_DICT[self.seq[i]]] = 1
        return one_hot_code

class peptide_detected():
    """
    Class for peptide detection.

    Attributes:
        models (dict): Dictionary to store trained models.
        simple_model (object): Simple neural network model.
        advance_model (object): Advanced neural network model.
        loss_data (list): List to store training and test loss data.
        X_train (torch.Tensor): Training data features.
        y_train (torch.Tensor): Training data labels.
        X_test (torch.Tensor): Test data features.
        y_test (torch.Tensor): Test data labels.
        weight_fn (float): Weight for positive samples in loss function.
        weight_fp (float): Weight for negative samples in loss function.
    """

    def __init__(self, path_to_pos_sample, path_to_neg_sample):
        """
        Initialize the peptide detector.

        Args:
            path_to_pos_sample (str): Path to file containing positive samples.
            path_to_neg_sample (str): Path to file containing negative samples.
        """
        self.models = {}
        self.simple_model = None
        self.advance_model = None
        self.loss_data = None
        self.X_train, self.y_train, self.X_test, self.y_test = \
            self.process_data(path_to_pos_sample, path_to_neg_sample)
        all_samples = (self.y_train == 1).sum().item() + (self.y_train == 0).sum().item()
        self.weight_fn = (self.y_train == 1).sum().item() / all_samples
        self.weight_fp = (self.y_train == 0).sum().item() / all_samples
        self.threshold= 0.5



    def read_file(self, pos_path: str, neg_path: str) -> Tuple[np.array, np.array, np.array, np.array]:
        """
        Read peptide data from files and process them.

        Args:
            pos_path (str): Path to file containing positive samples.
            neg_path (str): Path to file containing negative samples.

        Returns:
            Tuple: Tuple containing training and test data features and labels.
        """

        def read_data_from_file(file_name, label, length=9):
            """
            Read data from a file and create k-mer instances.

            Args:
                file_name (str): Name of the file.
                label (int): Label for the samples (0 for negative, 1 for positive).
                length (int): Length of k-mer.

            Returns:
                List: List of k-mer instances.
            """
            try:
                with open(file_name, "r") as file:
                    lines = file.readlines()
                    lines = [line.strip() for line in lines if line.strip()]
                    k_mers = []
                    for line in lines:
                        k_mers.append(K_mer(length, line, label))
                    return k_mers

            except Exception as e:
                print("can't open file. An error occurred:", e)
                raise Exception(e)

        def make_matrix(data):
            """
                        Create feature and label matrices from k-mer instances.

                        Args:
                            data (List): List of k-mer instances.

                        Returns:
                            Tuple: Tuple containing feature and label matrices.
                        """
            return pd.DataFrame([peptide.vector for peptide in data]).values, pd.DataFrame([peptide.label for peptide in data]).values

        # Call the function to read data from file
        pos_peptides = read_data_from_file(pos_path, 1)
        neg_peptides = read_data_from_file(neg_path, 0)

        # Take 90% of the data randomly
        train_data = random.sample(pos_peptides, k=int(len(pos_peptides) * 0.9)) + \
                     random.sample(neg_peptides, k=int(len(neg_peptides) * 0.9))

        # Get the remaining 10% of the data
        test_data = [peptide for peptide in pos_peptides if peptide not in train_data] \
                    + [peptide for peptide in neg_peptides if peptide not in train_data]
        train_data_X, train_data_y = make_matrix(train_data)
        test_data_X,  test_data_y = make_matrix(test_data)
        return train_data_X, train_data_y, test_data_X,  test_data_y


    def process_data(self, pos_path, neg_path ):
        """
        Process the peptide data from the given paths.

        Args:
            pos_path (str): Path to file containing positive samples.
            neg_path (str): Path to file containing negative samples.

        Returns:
            Tuple: A tuple containing torch.FloatTensor of training and test data features and labels.
        """
        train_data_X, train_data_y, test_data_X, test_data_y = self.read_file(pos_path, neg_path)
        X_train = torch.FloatTensor(train_data_X)
        X_test = torch.FloatTensor(test_data_X)
        y_train = torch.FloatTensor(train_data_y)  # Assuming y_train is binary (0 or 1)
        y_test = torch.FloatTensor(test_data_y)  # Assuming y_test is binary (0 or 1)
        return X_train, y_train, X_test, y_test

    def create_model(self, model:str, layer_1= None, layer_2= None ,linear = False):
        """
            Create a neural network model.

            Args:
                model (str): Type of model to create.
                layer_1 (int): Number of nodes in the first hidden layer.
                layer_2 (int): Number of nodes in the second hidden layer.
                linear (bool): Flag indicating whether the model is linear.

            Returns:
                None
            """
        if model == "SimpleNN":
            self.models["SimpleNN"] = snn.SimpleNN(input_size=180, hidden_size_1=180,hidden_size_2=180, weight_fn=self.weight_fn, weight_fp=self.weight_fp)
        if model == "advance_model":
            self.models["advance_model"] = snn.SimpleNN(input_size=180,hidden_size_1=layer_1, hidden_size_2=layer_2, weight_fn=self.weight_fn, weight_fp=self.weight_fp)
        if model == "linear_model":
            self.models["linear_model"] = snn.SimpleNN(input_size=180,hidden_size_1=layer_1, hidden_size_2=layer_2, weight_fn=self.weight_fn, weight_fp=self.weight_fp)

    def train_model(self, model:str, epoch:int):
        """
           Train the specified model.

           Args:
               model (str): Type of model to train.
               epoch (int): Number of epochs for training.

           Returns:
               None
           """
        self.models[model].train_model(self.X_train, self.y_train,
                                       self.X_test, self.y_test,
                                       epoch, self.loss_data)


    def test_model(self, model: str, threshold, insert_train:bool = False):
        """
            Test the specified model.

            Args:
                model (str): Type of model to test.
                threshold: Threshold value for classification.
                insert_train (bool): Flag indicating whether to test on the training set.

            Returns:
                Tuple: A tuple containing test results.
            """
        if insert_train:
            return self.models[model].test_model(self.X_train, self.y_train,
                                                 threshold)
        return self.models[model].test_model(self.X_test, self.y_test, threshold)

    def plot_loss(self, model):
        """
            Plot the training and test loss over epochs for the specified model.

            Args:
                model: Name of the model.

            Returns:
                None
            """
        train_loss = self.loss_data[0]
        test_loss = self.loss_data[1]

        # Create the figure and axis objects explicitly
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(len(train_loss)), train_loss, label='Training Loss')
        ax.plot(range(len(test_loss)), test_loss, label='Test Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Training and Test Loss over Epochs - {model}')
        ax.legend()
        ax.grid(True)

        # Show the plot
        plt.show()

        # Prompt user to save the plot
        save = input("Save plot? (y/n): ")
        if save.lower() == 'y':
            name = input("Add plot name: ")
            fig.savefig(f"./plots/{name} - {model}.png")

    def plot_roc_curve(self, model):
        """
            Plot the Receiver Operating Characteristic (ROC) curve for the specified model.

            Args:
                model: Name of the model.

            Returns:
                None
            """
        fpr, tpr, thresholds, roc_auc = self.models[model].roc_carve(self.X_test, self.y_test)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

        # Adding threshold markers
        for i, threshold in enumerate(np.arange(0.0, 1.1, 0.1)):
            idx = np.argmin(np.abs(thresholds - threshold))
            plt.scatter(fpr[idx], tpr[idx], marker='o', color='red')
            plt.text(fpr[idx], tpr[idx], f'{threshold:.2f}', fontsize=9,
                     ha='right')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()


def make_models():
    """
        Main function to train and test different models based on user input.

        Returns:
            None
        """
    try_again = True
    detector = None
    user_in = input("Do you want to train a new 180_model? (y/n)")
    if user_in == "y":
        while (try_again):
            if not detector:
                detector = peptide_detected(POS_PATH, NEG_PATH)
            detector.loss_data = [[], []]
            detector.create_model("SimpleNN")
            n_epoch: int = int(input("insert number of epochs? (int)"))
            detector.train_model("SimpleNN", n_epoch)
            select_threshold(detector, "SimpleNN")
            user_in = input("plot loss? (y/n)")
            if user_in == "y":

                detector.plot_loss("SimpleNN")

            user_in = input("save model? (y/n)")
            if user_in == "y":
                name = input("select model file name:")
                torch.save(detector.models['SimpleNN'], f"./saved_models/SimpleNN_{name}.pth")

            try_again = ask_for_another_try()

    try_again = True

    user_in = input("Do you want to train a new modular layers model? (y/n)")
    if user_in == "y":
        while (try_again):
            if not detector:
                detector = peptide_detected(POS_PATH, NEG_PATH)
            detector.loss_data = [[], []]
            layer_1 = int(input("how many nodes in layer 1?"))
            layer_2 = int(input("how many nodes in layer 2?"))
            detector.create_model("advance_model", layer_1= layer_1, layer_2= layer_2)
            n_epoch: int = int(input("insert number of epochs? (int)"))
            detector.train_model("advance_model", n_epoch)
            select_threshold(detector, "advance_model")
            user_in = input("plot loss? (y/n)")
            if user_in == "y":
                detector.plot_loss("advance_model")

            user_in = input("save model? (y/n)")
            if user_in == "y":
                name = input("select model file name:")
                torch.save(detector.models["advance_model"], f"saved_models/advanced_model_{name}.pth")
            try_again = ask_for_another_try()

    try_again = True
    user_in = input("Do you want to train a linear model? (y/n)")
    if user_in == "y":
        while (try_again):
            if not detector:
                detector = peptide_detected(POS_PATH, NEG_PATH)
            detector.loss_data = [[], []]
            layer_1 = int(input("how many nodes in layer 1?"))
            layer_2 = int(input("how many nodes in layer 2?"))
            detector.create_model("linear_model", layer_1=layer_1, layer_2=layer_2)
            n_epoch: int = int(input("insert number of epochs? (int)"))
            detector.train_model("linear_model", n_epoch)
            select_threshold(detector, "linear_model")
            user_in = input("plot loss? (y/n)")
            if user_in == "y":
                detector.plot_loss("advance_model")

            user_in = input("save model? (y/n)")
            if user_in == "y":
                name = input("select model file name:")
                torch.save(detector.models['linear_model'], f"saved_models/linear_model_{name}.pth")
            try_again = ask_for_another_try()
    input_user = input("Check models on Corona? (y/n)")

    if input_user == 'y':
        for model in detector.models.keys():
            detect(CORONA_SEQ, detector.models[model])

def ask_for_another_try():
    """
        Prompt the user to try the model training process again.

        Returns:
            bool: True if the user wants to try again, False otherwise.
        """

    try_again = input("Another try? (y/n)")
    if try_again == 'y':
        try_again = True
    else:
        try_again = False
    return try_again

def select_threshold(detector: peptide_detected, model):
    """
        Select the optimal threshold for the specified model.

        Args:
            detector (peptide_detected): Instance of the peptide detection class.
            model: Name of the model.

        Returns:
            None
        """
    max_acc = 0
    T = 0
    acc_l = []
    T_l = []
    for t in np.linspace(0.1, 1.0, 150):
        tn, fp, fn, tp, acc = detector.test_model(model, t, insert_train=True)
        acc_l.append(acc)
        T_l.append(t)
        if acc > max_acc:
            max_acc = acc
            T = t
    tn, fp, fn, tp, acc = detector.test_model(model, T)
    detector.threshold = T
    print(f'Total number of positive samples: {tp + fn}, number of false negative mistakes {fn} \n'
          f'Total number of negative samples: {tn + fp} , number of true negative mistakes {fp}\n'
          f'accuracy = {acc}')

CORONA_SEQ = "spike.txt"
def detect(seq_path: str, model: snn.SimpleNN):
    with open(seq_path) as f:
       seq = f.read().replace('\n', "")
    top_3 = [0, 0 ,0]
    top_3_seq = ["","",""]
    for i in range(len(seq)-9):
        k_mer = K_mer(9, seq[i:i+9])
        y_pred = float(model(torch.FloatTensor(k_mer.vector)))
        if y_pred > min(top_3):
            i = np.argmin(top_3)
            top_3[i] = y_pred
            top_3_seq[i] = k_mer.seq
    print(top_3_seq, top_3)



if __name__ == '__main__':
    make_models()






