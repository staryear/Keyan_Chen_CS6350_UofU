# This is a sample Python script.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
import random


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def convert_num_label(data_set):
    cur_class_column = data_set.get("label")
    temp_class = cur_class_column.copy(deep=True)
    temp_class.where(cur_class_column == 1, -1, inplace=True)
    temp_class.where(cur_class_column == 0, 1, inplace=True)
    data_set["label"] = temp_class
    return data_set


def SVM_primal(train_data, test_data, gamma, a, lr_mode):
    train_data_num = len(train_data)
    C_set = [100 / 873, 500 / 873, 700 / 873]
    train_data["bias"] = 1
    test_data["bias"] = 1
    weights = np.zeros(len(train_data.columns) - 1)
    print("when gamma = " + str(gamma) + " and when a = " + str(a))
    if lr_mode:
        print("when using problem (a) schedule of learning rate")
    else:
        print("when using problem (b) schedule")
    lr = 0
    for c in C_set:
        for t in range(0, 100):
            train_data = train_data.sample(frac=1, replace=False).reset_index(drop=True)  # shuffle
            for index, row in train_data.iterrows():
                row_features = row.drop("label")
                if lr_mode:
                    lr = gamma / (1 + gamma * t / a)
                else:
                    lr = gamma / (1 + t)
                if max(0, 1 - row.label * np.inner(weights, row_features)) == 0:
                    weights[:-1] = (1 - lr) * weights[:-1]
                else:
                    weight_without_bias = weights
                    weight_without_bias[-1] = 0
                    weights = weights - lr * weight_without_bias + c * train_data_num * lr * row.label * row_features.values
        features_test = test_data.drop("label", axis=1)
        test_prediction = np.sign(np.inner(weights, features_test))
        result = ((test_prediction - test_data["label"]) / 2).replace(-1, 1)
        average_error = np.sum(result) / len(test_data)

        features_train = train_data.drop("label", axis=1)
        train_prediction = np.sign(np.inner(weights, features_train))
        train_result = ((train_prediction - train_data["label"]) / 2).replace(-1, 1)
        train_average_error = np.sum(train_result) / len(train_data)

        print("when C = " + str(c) + " SVM train error is", train_average_error)
        print("when C = " + str(c) + " SVM test error is", average_error)
        print("when C = " + str(c) + " final weights are " + str(weights))


# Press the green button in the gutter to run the script.

def function_optim(alphas, parameter):
    return .5 * np.dot(np.dot(alphas.T, parameter), alphas) - np.sum(alphas)


def SVM_dual(train_data, test_data):
    C_set = [100 / 873, 500 / 873, 700 / 873]
    y = train_data.label * 1.0
    x = train_data.drop("label", axis=1) * 1.0

    parameter = np.dot(x.mul(y, axis=0), x.mul(y, axis=0).T)

    constraints = optimize.LinearConstraint(A=y, lb=0, ub=0)
    test_data["bias"] = 1

    for c in C_set:
        bound = optimize.Bounds(0, c)
        output = optimize.minimize(fun=function_optim, x0=np.zeros(len(train_data)), method="SLSQP",
                                   constraints=constraints,
                                   bounds=bound, args=(parameter))
        alphas = output.x
        weights = ((y * alphas).T @ x)
        Sup_Vec = (alphas > 1e-4).flatten()
        b = y[Sup_Vec] - np.dot(x[Sup_Vec], weights)

        weight_without_bias = weights.to_numpy()
        weights = np.append(weight_without_bias, b.values[0])

        features_test = test_data.drop("label", axis=1)
        test_prediction = np.sign(np.inner(weights, features_test))
        result = ((test_prediction - test_data["label"]) / 2).replace(-1, 1)
        average_error = np.sum(result) / len(test_data)

        features_train = train_data.drop("label", axis=1)
        features_train["bias"] = 1
        train_prediction = np.sign(np.inner(weights, features_train))
        train_result = ((train_prediction - train_data["label"]) / 2).replace(-1, 1)
        train_average_error = np.sum(train_result) / len(train_data)
        print("when C = " + str(c) + " SVM dual train error is", train_average_error)
        print("when C = " + str(c) + " SVM dual test error is " + str(average_error))
        print("when C = " + str(c) + " final weights are " + str(weights))


def gauss_distru(x, y, gamma):
    return np.exp(-gamma*np.linalg.norm(x - y) ** 2)

def SVM_Gauss(train_data, test_data, gamma, report_c):
    C_set = [100 / 873, 500 / 873, 700 / 873]
    y = train_data.label * 1.0
    x = train_data.drop("label", axis=1) * 1.0
    m, n = train_data.shape
    print(m, n)
    X_gauss = np.array([[gauss_distru(x.iloc[x1], x.iloc[x2], gamma) for x1 in range(m)] for x2 in range(m)])
    Y_gauss = np.outer(y, y)
    parameter = Y_gauss * X_gauss
    constraints = optimize.LinearConstraint(A=y, lb=0, ub=0)
    test_num = len(test_data)
    train_num = len(train_data)
    print("when gamma is " + str(gamma))
    for c in C_set:
        bound = optimize.Bounds(0, c)
        output = optimize.minimize(fun=function_optim, x0=np.zeros(len(train_data)), method="SLSQP",
                                   constraints=constraints,
                                   bounds=bound, args=(parameter))
        alphas = output.x
        Sup_Vec = (alphas > 1e-4).flatten()
        weights = ((y * alphas).T @ x)
        features_test = test_data.drop("label", axis=1)
        SV_prediction =  np.array([[gauss_distru(features_test.iloc[x1], sv, gamma)
                    for i, sv in x[Sup_Vec].iterrows()] for x1 in range(test_num)])
        print(len(SV_prediction))
        test_prediction = np.sign(np.inner(SV_prediction, alphas[Sup_Vec] * y[Sup_Vec]))
        test_prediction = np.sign(np.inner(weights, features_test))
        result = ((test_prediction - test_data["label"]) / 2).replace(-1, 1)
        average_error = np.sum(result) / len(test_data)

        features_train = train_data.drop("label", axis=1)
        SV_train_prediction = np.array([[gauss_distru(features_train.iloc[x1], sv, gamma)
                                   for i, sv in x[Sup_Vec].iterrows()] for x1 in range(train_num)])
        train_prediction = np.sign(np.inner(SV_train_prediction, alphas[Sup_Vec] * y[Sup_Vec]))
        train_prediction = np.sign(np.inner(weights, features_train))
        train_result = ((train_prediction - train_data["label"]) / 2).replace(-1, 1)
        train_average_error = np.sum(train_result) / len(train_data)
        print("when C = " + str(c) + " SVM dual Gauss train error is " + str(train_average_error))
        print("when C = " + str(c) + " SVM dual Gauss test error is " + str(average_error))

        if report_c:
            print("when C = " + str(c) + " the number of support vector " + str(len(alphas[alphas > 1e-4])))
            if c == 500/873:
                print("when gamma is " + str(gamma) + " and the c = 500/873, the support vector is \n" + str(x[Sup_Vec]))



if __name__ == '__main__':
    bank_note_columns = ["variance", "skewness", "curtosis", "entropy", "label"]
    train_data_bn = pd.read_csv("./bank-note/train.csv", names=bank_note_columns, header=None)
    test_data_bn = pd.read_csv("./bank-note/test.csv", names=bank_note_columns, header=None)
    train_data_bn = convert_num_label(train_data_bn)
    test_data_bn = convert_num_label(test_data_bn)
    print(train_data_bn.head(5))
    gamma = 0.5
    a = 0.5
    lr_mode = True
    SVM_primal(train_data_bn, test_data_bn, gamma, a, lr_mode)
    gamma = 1
    lr_mode = False
    SVM_primal(train_data_bn, test_data_bn, gamma, a, lr_mode)
    SVM_dual(train_data_bn, test_data_bn)
    report_c = False
    gamma_list = [0.1, 0.5, 1, 5, 100]
    for gamma in gamma_list:
        SVM_Gauss(train_data_bn, test_data_bn, gamma, report_c)
    report_c = True
    for gamma in gamma_list:
        SVM_Gauss(train_data_bn, test_data_bn, gamma, report_c)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
