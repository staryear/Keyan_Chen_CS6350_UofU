# This is a sample Python script.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def perceptron(train_data, test_data, LR):
    features = train_data.drop("label", axis=1)
    print(features)
    weights = [0] * len(features.columns)
    print(weights)
    for t in range(0, 10):
        re_sample = train_data.sample(frac=1, replace=False).reset_index(drop=True)
        #re_sample = train_data
        for i in range(len(re_sample)):
            row = re_sample.iloc[i, :]
            row_features = row.drop("label")
            #print(row_features)
            prediction = np.sign(np.inner(weights, row_features))
            if prediction != row["label"]:
                weights = weights + LR * np.inner(row["label"], row_features)
    features_test = test_data.drop("label", axis=1)
    test_prediction = np.sign(np.inner(weights, features_test))
    result = ((test_prediction - test_data["label"])/2).replace(-1, 1)
    average_error = np.sum(result)/len(test_data)

    return average_error, weights

def voted_perceptron(train_data, test_data, LR):
    features = train_data.drop("label", axis=1)
    print(features)
    weights = [0] * len(features.columns)
    vote_weights_list = [[weights]]
    vote = [0]
    m = 0
    for t in range(0, 10):
        re_sample = train_data.sample(frac=1, replace=False).reset_index(drop=True)
        #re_sample = train_data
        for i in range(len(re_sample)):
            row = re_sample.iloc[i, :]
            row_features = row.drop("label")
            #print(row_features)
            prediction = np.sign(np.inner(weights, row_features))
            if prediction != row["label"]:
                weights = weights + LR * np.inner(row["label"], row_features)
                vote_weights_list.append([weights])
                m += 1
                vote.append(1)
            else:
                vote[m] += 1



    features_test = test_data.drop("label", axis=1)
    test_prediction = np.sign(np.inner(vote_weights_list, features_test)[:,0])
    prediction_result = np.sign(np.sum(pd.DataFrame(test_prediction).multiply(vote, axis='rows')))
    result = ((prediction_result - test_data["label"]) / 2).replace(-1, 1)
    average_error = np.sum(result) / len(test_data)

    return average_error, vote_weights_list, vote

def average_perceptron(train_data, test_data, LR):
    features = train_data.drop("label", axis=1)
    print(features)
    weights = [0] * len(features.columns)
    print(weights)
    ave = [0] * len(features.columns)
    for t in range(0, 10):
        re_sample = train_data.sample(frac=1, replace=False).reset_index(drop=True)
        #re_sample = train_data
        for i in range(len(re_sample)):
            row = re_sample.iloc[i, :]
            row_features = row.drop("label")
            # print(row_features)
            prediction = np.sign(np.inner(weights, row_features))
            if prediction != row["label"]:
                weights = weights + LR * np.inner(row["label"], row_features)
            ave += weights
    features_test = test_data.drop("label", axis=1)
    test_prediction = np.sign(np.inner(ave, features_test))
    result = ((test_prediction - test_data["label"]) / 2).replace(-1, 1)
    average_error = np.sum(result) / len(test_data)

    return average_error, weights

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    bank_note_columns = ["variance","skewness","curtosis", "entropy", "label"]
    train_data_bn = pd.read_csv("./bank-note/train.csv", names=bank_note_columns, header=None)
    test_data_bn = pd.read_csv("./bank-note/test.csv", names=bank_note_columns, header=None)
    train_data_bn = convert_num_label(train_data_bn)
    test_data_bn = convert_num_label(test_data_bn)
    print(train_data_bn.head(5))

    average_perceptron_error, learned_weight = perceptron(train_data_bn, test_data_bn, 0.01)
    print("standard perceptron error", average_perceptron_error)
    print("final weights of perceptron", learned_weight)

    average_voted_perceptron_error, voted_learned_weight, counts = voted_perceptron(train_data_bn, test_data_bn, 0.01)
    print("voted perceptron error", average_voted_perceptron_error)
    print("final weights of voted perceptron", voted_learned_weight)
    print("counts list of voted perceptron", counts)

    average_ave_perceptron_error, ave_learned_weight = average_perceptron(train_data_bn, test_data_bn, 0.01)
    print("average perceptron error", average_ave_perceptron_error)
    print("final weights of average perceptron", ave_learned_weight)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
