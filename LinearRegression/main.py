# This is a sample Python script.
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def gradient(weight, dataset, label_name):
    features = dataset.drop(label_name, axis=1)
    grad_list = []
    for i in range(0, len(features.columns)):
        result = -np.sum((dataset[label_name] - np.inner(weight, features))*features.iloc[:, i])
        grad_list.append(result)

    return grad_list
def LMSRegression(dataset, rate, threshold, label_name):

    features = dataset.drop(label_name, axis = 1)
    #print(features)
    weight = np.zeros(len(features.columns))
    print("init weight: " + str(weight))
    cost_function = []

    norm = np.inf
    while norm > threshold:
        grad = pd.Series(gradient(weight, dataset,label_name ))
        #print(grad)
        updated_weight = weight - rate * grad
        cost_result = 0.5 * np.sum((dataset[label_name] - np.inner(weight, features))**2)
        cost_function.append(cost_result)
        norm = np.linalg.norm(updated_weight - weight)
        weight = updated_weight
        #print(norm)

    #print(np.asarray(weight), rate)
    return weight, cost_function

def lms_SGD(dataset, rate, threshold, label_name):
    features = dataset.drop(label_name, axis = 1)
    #print(features)
    weight = np.zeros(len(features.columns))
    print("init weight: " + str(weight))
    cost_function = []

    norm = np.inf
    step = 0
    while norm > threshold or len(cost_function) < 2:
        row_info =dataset.iloc[(step%len(dataset)), :]
        single_features = row_info.drop(label_name)
        grad_result = (row_info[label_name] - np.inner(weight, single_features)) * single_features
        #print(grad_result)
        updated_weight = weight + rate * grad_result
        cost_result = 0.5 * np.sum((dataset[label_name] - np.inner(weight, features))**2)
        cost_function.append(cost_result)
        if len(cost_function) >= 2:
            norm = np.abs(cost_function[-1] - cost_function[-2])
        weight = updated_weight
        step += 1
        #print(norm)

    #print(np.asarray(weight), rate)
    return weight, cost_function
def ploting(line1,  label1, title):
    plt.plot(np.asarray(line1), label=label1)
    plt.title(title)
    plt.xlabel("steps")
    plt.ylabel("cost function")
    plt.legend()
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    concrete_columns = ["Cement", "Slag", "Fly ash", "Water", "SP", "Coarse Aggr", "Fine Aggr", "SLUMP"]
    train_data_con = pd.read_csv("./concrete/train.csv", names=concrete_columns, header=None)
    test_data_con = pd.read_csv("./concrete/test.csv", names=concrete_columns, header=None)

    rate = 0.015
    weight, cost_list = LMSRegression(train_data_con, rate, 10**-6, "SLUMP")
    print("under learning rate " + str(rate) + " learned weights are " + str(np.asarray(weight)))
    #print(cost_list)
    test_features = test_data_con.drop("SLUMP", axis=1)
    test_cost_result = 0.5 * np.sum((test_data_con["SLUMP"] - np.inner(weight, test_features)) ** 2)
    print("testdata cost", test_cost_result)
    ploting(cost_list, "cost", "cost function via steps")

    rate = 0.03
    weight, cost_list = lms_SGD(train_data_con, rate, 10 ** -6, "SLUMP")
    print("under learning rate " + str(rate) + " learned weights are " + str(np.asarray(weight)))
    #print(cost_list)
    test_features = test_data_con.drop("SLUMP", axis=1)
    test_cost_result = 0.5 * np.sum((test_data_con["SLUMP"] - np.inner(weight, test_features)) ** 2)
    print("testdata cost", test_cost_result)
    ploting(cost_list, "cost", "cost function via steps")




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
