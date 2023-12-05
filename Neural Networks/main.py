# This is a sample Python script.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
import random
import torch
import torch.nn as nn
import itertools as it
import os

#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def convert_num_label(data_set):
    cur_class_column = data_set.get("label")
    temp_class = cur_class_column.copy(deep=True)
    temp_class.where(cur_class_column == 1, -1, inplace=True)
    temp_class.where(cur_class_column == 0, 1, inplace=True)
    data_set["label"] = temp_class
    return data_set


class NeuralNetwork:
    def __init__(self, n1, n2, X_train, zero_weights):
        self.n1 = n1
        self.n2 = n2
        self.w_1 = np.random.normal(0, 1, size=(X_train.shape[1], self.n1))
        self.w_2 = np.random.normal(0, 1, size=(self.n1 + 1, self.n2))
        self.w_3 = np.random.normal(0, 1, size=(self.n2 + 1, 1))
        if zero_weights:
            self.w_1 = np.zeros((X_train.shape[1], self.n1))
            self.w_2 = np.zeros((self.n1 + 1, self.n2))
            self.w_3 = np.zeros((self.n2 + 1, 1))


    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def der_sigmoid(self, output):
        return output * (1 - output)

    def loss(self, y, pred_y):
        return (1 / 2) * (pred_y - y) ** 2

    def forward(self, x, y):
        self.o1 = np.append(1, self.sigmoid(np.dot(x, self.w_1)))
        self.o2 = np.append(1, self.sigmoid(np.dot(self.o1, self.w_2)))
        self.o3 = np.dot(self.o2, self.w_3)
        self.los = self.loss(y, self.o3)

        return self.o3, self.los

    def backward(self, output, x, y):
        self.grad_3 = (output - y) * self.o2
        temp = (output - y) * self.w_3[1:].T * self.der_sigmoid(self.o2[1:])
        self.grad_2 = (np.repeat(temp, self.n1 + 1, 0).T * self.o1).T
        temp = np.sum(np.dot((output - y) * self.w_3[1:] * self.der_sigmoid(self.o2[1:]), self.w_2[1:].T),
                      axis=0).reshape(1, -1)
        self.grad_1 = (np.repeat(temp, len(x), 0) * self.der_sigmoid(self.o1[1:])).T * x
        return self.grad_3, self.grad_2, self.grad_1.T

    def fit(self, train_data, X_train, Y_train, X_test, Y_test, lr=0.01, epochs=10):

        test_pred = []
        train_pred = []
        train_data = train_data.sample(frac=1).reset_index(drop=True)
        X = train_data.iloc[:, :-1]
        Y = train_data.iloc[:, -1]
        for epoch in range(epochs):
            ans_loss = []
            for i in range(len(X)):
                output, lo = self.forward(X.iloc[i].values, Y[i])
                ans_loss.append(lo[0])
                g_3, g_2, g_1 = self.backward(output, X.iloc[i].values, Y[i])
                self.w_1 = self.w_1 - lr * g_1
                self.w_2 = self.w_2 - lr * g_2
                self.w_3 = (self.w_3.reshape(1, -1) - lr * g_3.reshape(1, -1)).reshape(-1, 1)
            lr = lr / (1 + ((lr / 0.5) * epoch))

        for x, y in zip(X_train, Y_train):
            pred, fin_loss = self.forward(x, y)
            train_pred.append(pred)
        for x, y in zip(X_test, Y_test):
            pred, fin_loss = self.forward(x, y)
            test_pred.append(pred)
        return train_pred, test_pred

def calculate_rate(prediction_result, label):
    count = 0

    for i in range(len(prediction_result)):
        if prediction_result[i][0] > 0.5:
            if label[i] == 1:
                count += 1
        else:
            if label[i] == 0:
                count += 1
    error_rate = (len(prediction_result) - count) / len(prediction_result)

    return error_rate

class Net(nn.Module):
    def __init__(self, layers, act=nn.Tanh()):
        super(Net, self).__init__()
        self.act = act
        self.fc = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.fc.append(nn.Linear(layers[i], layers[i + 1]))
            if act == nn.Tanh():
                nn.init.xavier_normal_(self.fc[-1].weight)
            else:
                nn.init.kaiming_uniform_(self.fc[-1].weight, nonlinearity="relu")


    def forward(self, x):
        for i in range(len(self.fc) - 1):
            x = self.fc[i](x)
            x = self.act(x)
        x = self.fc[-1](x)
        x = nn.functional.sigmoid(x)
        return x

def to_tensor(X):
    return torch.from_numpy(X).float()

def train(xtr,ytr,xte,yte,act_f, act_name):
    d = xtr.shape[1]
    depth = [3, 5, 9]
    for n in width:
        for deep in depth:
            layers = [d]
            for i in range(deep - 2):
                layers.append(n)
            layers.append(1)
            print("with width " + str(n) + " and depth " + str(deep) + " and act is "+ act_name +": ")
            print(layers)
            fnn = Net(layers,act_f)
            optimizer = torch.optim.Adam(fnn.parameters(), lr=1e-3)
            nepoch = 100
            loss_fn = nn.BCELoss()
            errors = 0
            tr_errors = 0
            for epoch in range(nepoch):
                pred = fnn(xtr)
                loss = loss_fn(pred, ytr.unsqueeze(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                tr_outputs = fnn(xtr)
                tr_predicted = np.where(tr_outputs < 0.5, 0, 1)
                tr_predicted = list(it.chain(*tr_predicted))
                tr_errors += abs(tr_predicted - ytr.numpy()).sum()
                tr_rate = tr_errors / len(ytr)

                outputs = fnn(xte)
                predicted = np.where(outputs < 0.5, 0, 1)
                predicted = list(it.chain(*predicted))
                errors += abs(predicted - yte.numpy()).sum()
                rate = errors / len(yte)
                print("train error_rate = ", tr_rate)
                print("test error_rate = ", rate)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    bank_note_columns = ["variance", "skewness", "curtosis", "entropy", "label"]
    train_data_bn = pd.read_csv("./bank-note/train.csv", names=bank_note_columns, header=None)
    test_data_bn = pd.read_csv("./bank-note/test.csv", names=bank_note_columns, header=None)
    #train_data_bn = convert_num_label(train_data_bn)
    #test_data_bn = convert_num_label(test_data_bn)
    print(train_data_bn.head(5))

    X = train_data_bn.iloc[:, :-1]
    Y_train = train_data_bn.iloc[:, -1]
    X_train = np.column_stack(([1] * X.shape[0], X))

    train_data = np.column_stack(([1] * train_data_bn.shape[0], train_data_bn))
    train_data = pd.DataFrame(train_data)
    X_test = test_data_bn.iloc[:, :-1]
    X_test = np.column_stack(([1] * X_test.shape[0], X_test))
    Y_test = test_data_bn.iloc[:, -1]

    width = [5, 10, 25, 50, 100]
    print("initial random weights")
    for n in width:
        nn = NeuralNetwork(n, n, X_train, False)
        p_tr, p_te = nn.fit(train_data, X_train, Y_train, X_test, Y_test, 0.01, 20)
        count = 0
        print("Number of width {}".format(n))
        train_error_rate = calculate_rate(p_tr, Y_train)
        print("random weights NN Train error is ", train_error_rate)

        test_error_rate = calculate_rate(p_te, Y_test)
        print("random weights NN Test error is ", test_error_rate)

    print("initial zero weights")
    for n in width:
        nn = NeuralNetwork(n, n, X_train, True)
        p_tr, p_te = nn.fit(train_data, X_train, Y_train, X_test, Y_test, 0.01, 100)
        count = 0
        print("Number of width {}".format(n))
        train_error_rate = calculate_rate(p_tr, Y_train)
        print("zero weights NN Train error is ", train_error_rate)

        test_error_rate = calculate_rate(p_te, Y_test)
        print("zero weights NN Test error is ", test_error_rate)

    print("pytorch part")
    xtr = train_data_bn.iloc[:, :-1]
    ytr = train_data_bn.iloc[:, -1]
    xte = test_data_bn.iloc[:, :-1]
    yte = test_data_bn.iloc[:, -1]
    xtr = to_tensor(xtr.to_numpy().astype(np.float32))
    ytr = to_tensor(ytr.to_numpy().astype(np.float32))
    xte = to_tensor(xte.to_numpy().astype(np.float32))
    yte = to_tensor(yte.to_numpy().astype(np.float32))
    act_f = nn.Tanh()
    train(xtr, ytr, xte, yte, act_f, "Tanh")
    act_f = nn.ReLU()
    train(xtr, ytr, xte, yte, act_f, "Relu")





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
