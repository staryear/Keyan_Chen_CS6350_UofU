# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


class MAP:
    def __init__(self, X, y, std):
        self.X = X
        # import pdb;pdb.set_trace()
        self.X = np.column_stack((X, np.ones([X.shape[0], 1])))
        self.y = y
        self.std = std
        self.w = np.zeros(X.shape[1] + 1)

    def loss(self, x, y, w):
        return np.log(1 + np.exp(-y * np.inner(w, x))) + 1 / (self.std ** 2) * np.inner(w, w)

    def grad_loss(self, x, y, w):
        return -sigmoid(-y * np.inner(w, x)) * (y * x) + 2 / self.std * w

    def pred(self, X):
        return np.sign(np.inner(self.w[0:-1], X) + self.w[-1])

    def train_error(self):
        return np.count_nonzero(np.sign(np.inner(self.w, self.X)).flatten() * self.y.flatten() != 1) / len(self.X)


class ML:
    def __init__(self, X, y, std):
        self.X = X
        self.X = np.column_stack((X, np.ones([X.shape[0], 1])))
        self.y = y
        self.w = np.zeros(X.shape[1] + 1)

    def loss(self, x, y, w):
        return np.log(1 + np.exp(-y * np.inner(w, x)))

    def grad_loss(self, x, y, w):
        return -sigmoid(-y * np.inner(w, x)) * (y * x)

    def pred(self, X):
        return np.sign(np.inner(self.w[0:-1], X) + self.w[-1])

    def train_error(self):
        return np.count_nonzero(np.sign(np.inner(self.w, self.X)).flatten() * self.y.flatten() != 1) / len(self.X)


def shuffle(X, y):
    assert len(X) == len(y)
    p = np.random.permutation(len(X))
    return X[p], y[p]


def gamma(t):
    gamma0 = 1e-2
    d = 1e-4
    return gamma0 / (1 + gamma0 / d * t)

def train_models(Xtr, ytr, Xte, yte, model_type):
    V = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
    epochs = 100
    for v in V:
        std = v ** 2
        if model_type == "map":
            MAP_model = MAP(Xtr, ytr, std)
        else:
            MAP_model = ML(Xtr, ytr, std)
        w = MAP_model.w
        error_hist = []

        for t in range(epochs):
            X = MAP_model.X
            Y = MAP_model.y
            p = np.random.permutation(X.shape[0])
            for x, y in zip(X[p], Y[p]):
                w = w - gamma(t) * MAP_model.grad_loss(x, y, w)
            MAP_model.w = w
            error_hist.append(MAP_model.train_error())
        MAP_model.w = w
        hte = MAP_model.pred(Xte)
        htr = MAP_model.pred(Xtr)
        map_test_error = np.count_nonzero(hte * yte.flatten() != 1) / len(yte)
        map_train_error = np.count_nonzero(htr * ytr.flatten() != 1) / len(ytr)
        print("when v is " + str(v) + " the " + model_type + " train error is " + str(map_train_error))
        print("and the test error is ", map_test_error)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    bank_note_columns = ["variance", "skewness", "curtosis", "entropy", "label"]
    data = pd.read_csv('./bank-note/train.csv')
    data_test = pd.read_csv('./bank-note/test.csv')
    data = pd.DataFrame(data.to_numpy(), columns=bank_note_columns)
    data_test = pd.DataFrame(data_test.to_numpy(), columns=bank_note_columns)
    print(data.head(5))
    data["label"] = data["label"] * 2 - 1
    data_test["label"] = data_test["label"] * 2 - 1
    Xtr = data[["variance", "skewness", "curtosis", "entropy"]].to_numpy()
    ytr = data["label"].to_numpy().reshape(-1, 1)
    Xte = data_test[["variance", "skewness", "curtosis", "entropy"]].to_numpy()
    yte = data_test["label"].to_numpy().reshape(-1, 1)

    train_models(Xtr, ytr, Xte, yte, "map")
    train_models(Xtr, ytr, Xte, yte, "ml")




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
