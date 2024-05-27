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
        self.models = {}
        self.simple_model = None
        self.advance_model = None
        self.loss_data = None
        self.X_train, self.y_train, self.X_test, self.y_test =\
            self.process_data(path_to_pos_sample,path_to_neg_sample)
        all_samples = (self.y_train == 1).sum().item() + (self.y_train == 0).sum().item()
        self.weight_fn = (self.y_train == 1).sum().item() / all_samples
        self.weight_fp = (self.y_train == 0).sum().item() / all_samples

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

        # Get the remaining 10% of the data
        test_data = [peptide for peptide in pos_peptides if peptide not in train_data] \
                    + [peptide for peptide in neg_peptides if peptide not in train_data]
        train_data_X, train_data_y = make_matrix(train_data)
        test_data_X,  test_data_y = make_matrix(test_data)
        return train_data_X, train_data_y, test_data_X,  test_data_y


    def process_data(self, pos_path, neg_path ):
        train_data_X, train_data_y, test_data_X, test_data_y = self.read_file(pos_path, neg_path)
        X_train = torch.FloatTensor(train_data_X)
        X_test = torch.FloatTensor(test_data_X)
        y_train = torch.FloatTensor(train_data_y)  # Assuming y_train is binary (0 or 1)
        y_test = torch.FloatTensor(test_data_y)  # Assuming y_test is binary (0 or 1)
        return X_train, y_train, X_test, y_test

    def create_model(self, model:str):
        if model == "SimpleNN":
            self.models["SimpleNN"] = snn.SimpleNN(input_size=180, hidden_size_1=180,hidden_size_2=180, weight_fn=self.weight_fn, weight_fp=self.weight_fp)
            # self.simple_model = snn.SimpleNN(180, 180, weight_fn=self.weight_fn, weight_fp=self.weight_fp)
        if model == "advance_model":
            # self.advance_model = snn.SimpleNN(180, 9, weight_fn=self.weight_fn, weight_fp=self.weight_fp)
            self.models["advance_model"] = snn.SimpleNN(input_size=180,hidden_size_1=20, hidden_size_2=9, weight_fn=self.weight_fn, weight_fp=self.weight_fp)


    def train_model(self, model:str, epoch:int):
        self.models[model].train_model(self.X_train, self.y_train,
                                          self.X_test, self.y_test,
                                          epoch, self.loss_data)


    def test_model(self, model: str, threshold, insert_train:bool = False):
        if insert_train:
            return self.models[model].test_model(self.X_train, self.y_train,
                                                 threshold)
        return self.models[model].test_model(self.X_test, self.y_test, threshold)

    def plot_loss(self, model):
        train_loss = self.loss_data[0]
        test_loss = self.loss_data[1]

        # Plotting the training and test losses
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(train_loss)), train_loss, label='Training Loss')
        plt.plot(range(len(test_loss)), test_loss, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Test Loss over Epochs- {model}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"./plots/Training and Test Loss over Epochs- {model}.png")

    def plot_roc_curve(self, model):
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


if __name__ == '__main__':
    pos_path = "pos_A0201.txt"
    neg_path = "neg_A0201.txt"
    # p_detected = peptide_detected(pos_path, neg_path)
    for model, n_epoch in zip(["SimpleNN", "advance_model"], [30, 40]):
        p_detected = peptide_detected(pos_path, neg_path)
        p_detected.loss_data = [[], []]
        p_detected.create_model(model)
        p_detected.train_model(model, 100)
        p_detected.plot_loss(model)
        p_detected = peptide_detected(pos_path, neg_path)
        p_detected.create_model(model)
        p_detected.train_model(model, n_epoch)
        max_acc = 0
        T = 0
        acc_l = []
        T_l = []
        for t in np.linspace(0.1, 1.0, 150):
            tn, fp, fn, tp, acc = p_detected.test_model(model, t, insert_train=True)
            acc_l.append(acc)
            T_l.append(t)
            # print(f't = {t}. accurachy = {acc}')
            if acc > max_acc:
                max_acc = acc
                T = t
        print(f'########  for model {model} with {n_epoch} epoch and T = {T}  #######')
        tn, fp, fn, tp, acc = p_detected.test_model(model, T)

        print(f'from numer_of_pos: {tp + fn} wrong on {fn} \n'
              f'from numer_of_neg: {tn + fp} wrong on {fp}\n'
              f'accurachy = {acc}')

    # p_detected.loss_data = [[], []]
    # p_detected.create_model("SimpleNN")
    # p_detected.train_model("SimpleNN", 30)
    # p_detected.plot_loss("SimpleNN")
    #
    #
    # p_detected.plot_roc_curve()
    #
    # tn, fp, fn, tp, acc = p_detected.test_model("SimpleNN", 0.5)
    #
    # print (f'from numer_of_pos: {tp + fn} wrong on {fn} \n'
    #        f'from numer_of_neg: {tn + fp} wrong on {fp}\n'
    #        f'ccurachy = {acc}')
    #
    # print("####################################################################")
    #
    # p_detected.create_model("advance_model")
    # p_detected.train_model("advance_model", 40)
    # p_detected.plot_loss("advance_model")
    # max_acc = 0
    # T = 0
    # acc_l = []
    # T_l = []
    # for t in np.linspace(0.1, 1.0, 150):
    #     tn, fp, fn, tp, acc = p_detected.test_model("advance_model", t, insert_train=True)
    #     acc_l.append(acc)
    #     T_l.append(t)
    #     print(f't = {t}. accurachy = {acc}')
    #     if acc > max_acc:
    #         max_acc = acc
    #         T=t

    # Plotting the graph of threshold vs accuracy
    # plt.figure(figsize=(10, 6))
    # plt.plot(T_l, acc_l, marker='o')
    # plt.title('Threshold vs Accuracy')
    # plt.xlabel('Threshold (t)')
    # plt.ylabel('Accuracy')
    # plt.grid(True)
    # plt.show()

    # print(f'T = {T}')
    # tn, fp, fn, tp, acc = p_detected.test_model("advance_model", T)
    #
    # print (f'from numer_of_pos: {tp + fn} wrong on {fn} \n'
    #        f'from numer_of_neg: {tn + fp} wrong on {fp}\n'
    #        f'accurachy = {acc}')





