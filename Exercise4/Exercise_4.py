from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from Exercise3.DatasetGeneration import generate_dataset, get_probabilities, read_dataset, restourant_path
import numpy as np
import pandas as pd


def load_train_test(percent=80, dataframe=None):
    ground_truth = read_dataset(hidden_value=-1, data=dataframe)

    X = []
    Y = []
    for entry in ground_truth:
        # if entry[1] is None:
        #     entry[1] = -1
        # if entry[2] is None:
        #     entry[2] = -1
        if entry[1] is None and entry[2] is None:
            entry[1] = entry[0]
            entry[2] = entry[0]
        elif entry[1] is None:
            entry[1] = (entry[0] + entry[2]) * 0.5
        elif entry[2] is None:
            entry[2] = (entry[0] + entry[1]) * 0.5
        X.append([entry[0], entry[1], entry[2]])
        Y.append(entry[3])
    X = np.array(X)
    Y = np.array(Y)
    n = int(len(X) * percent * 0.01)
    X_train = X[2:n]  # Starts from 2 to remove the dataframe header (the column names)
    X_test = X[n:]
    Y_train = Y[2:n]  # Starts from 2 to remove the dataframe header (the column names)
    Y_test = Y[n:]
    return X_train, X_test, Y_train, Y_test


def compute_accuracy(data, model):
    gt = get_probabilities(data, True)
    correct = 0
    incorrect = 0
    for f in range(5):
        for s in [0, 1, 2, 3, 4, 5, None]:
            for v in [0, 1, 2, 3, 4, 5, None]:
                ns = s
                nv = v
                if s is None and v is None:
                    ns = f
                    nv = f
                elif s is None:
                    ns = (f + v) * 0.5
                elif v is None:
                    nv = (f + s) * 0.5
                result = model.predict(np.array([f, ns, nv]).reshape(1, -1))
                result = round(result[0])
                if result == np.argmax(gt[(f, s, v)]) + 1:
                    correct += 1
                else:
                    incorrect += 1
    return correct / (correct + incorrect)


def print_accuracy(message, score_train, score_test):
    print("Accuracy of " + message + ": ")
    print("Training: " + str(score_train))
    print("Test: " + str(score_test))


def get_partitions(model, printDebug=False):
    p1star = set()
    p2star = set()
    p3star = set()
    for f in range(5):
        for s in [0, 1, 2, 3, 4, 5, None]:
            for v in [0, 1, 2, 3, 4, 5, None]:
                tup = (f, s, v)
                ns = s
                nv = v
                if s is None and v is None:
                    ns = f
                    nv = f
                elif s is None:
                    ns = (f + v) * 0.5
                elif v is None:
                    nv = (f + s) * 0.5
                result = model.predict(np.array([f, ns, nv]).reshape(1, -1))
                if printDebug:
                    print("#################################")
                    print(model.coef_)
                    print(model.intercept_)
                result = round(result[0])
                if result <= 1:
                    p1star.add(tup)
                elif result >= 3:
                    p3star.add(tup)
                else:
                    p2star.add(tup)
    return p1star, p2star, p3star


def logistic_regression(X_train, X_test, Y_train, Y_test):
    clf = LogisticRegression(penalty='l1', solver='saga', random_state=0).fit(X_train, Y_train)
    return clf


def logistic_regression_forced(X_train, X_test, Y_train, Y_test):

    clf = LogisticRegression(penalty='l1', solver='saga', random_state=0).fit(X_train, Y_train)

    intercept = []
    for i in clf.intercept_:
        if i < 0:
            intercept.append(0)
        else:
            intercept.append(i)
    clf.intercept_ = np.array(intercept)

    coeff = []
    for row in clf.coef_:
        r = []
        for coef in row:
            if coef < 0:
                r.append(0)
            else:
                r.append(coef)
        coeff.append(r)
    clf.coef_ = np.array(coeff)

    return clf


def linear_regression(X_train, X_test, Y_train, Y_test):
    lin = LinearRegression().fit(X_train, Y_train)
    return lin


def linear_regression_forced(X_train, X_test, Y_train, Y_test):
    lin = LinearRegression(positive=True).fit(X_train, Y_train)
    return lin


def lasso_regression(X_train, X_test, Y_train, Y_test):
    las = Lasso(positive=True, alpha=0.1).fit(X_train, Y_train)
    return las


def ridge_regression(X_train, X_test, Y_train, Y_test):
    rid = Ridge(alpha=0.1).fit(X_train, Y_train)
    return rid


def run_exercise_4():
    accuracy = {}

    # Generate 100 different values for sigma (Standard deviation of the gaussian used to generate the dataset)
    dataset_variance = list(np.linspace(0.1, 0.5, 100))

    cnt = 0
    mmax = len(dataset_variance)

    for sigma in dataset_variance:
        cnt += 1
        print("Iteration: " + str(cnt) + "/" + str(mmax))
        # Generate dataset
        dataset = generate_dataset(10000, sigma)
        # Converts dataset to Pandas Dataframe
        df = pd.DataFrame(dataset, columns=['Food', 'Service', 'Value', 'Stars'])
        # Splits to create training and test set
        X, Y, x, y = load_train_test(80, df.values.tolist())
        # print(X)
        # print(Y)
        # print(x)
        # print(y)

        # Run all the methods on the dataset and evaluate the performances
        accuracy["Logistic_Normal"] = logistic_regression(X,Y,x,y)
        accuracy["Logistic_Forced"] = logistic_regression_forced(X,Y,x,y)
        accuracy["Linear_Normal"] = linear_regression(X,Y,x,y)
        accuracy["Linear_Forced"] = linear_regression_forced(X,Y,x,y)
        accuracy["Lasso"] = lasso_regression(X,Y,x,y)
        accuracy["Ridge"] = ridge_regression(X,Y,x,y)

    # Compute average accuracies:
    print("Analysis of accuracies: ")
    print("[Logistic Regression (Normal)]: Max: " + str(max(accuracy["Logistic_Normal"])))
    print("[Logistic Regression (Normal)]: Min: " + str(min(accuracy["Logistic_Normal"])))
    print("[Logistic Regression (Normal)]: Mean: " + str(np.mean(accuracy["Logistic_Normal"])))
    print("[Logistic Regression (Normal)]: Standard Deviation: " + str(np.std(accuracy["Logistic_Normal"])))

    print("[Logistic Regression (Forced)]: Max: " + str(max(accuracy["Logistic_Forced"])))
    print("[Logistic Regression (Forced)]: Min: " + str(min(accuracy["Logistic_Forced"])))
    print("[Logistic Regression (Forced)]: Mean: " + str(np.mean(accuracy["Logistic_Forced"])))
    print("[Logistic Regression (Forced)]: Standard Deviation: " + str(np.std(accuracy["Logistic_Forced"])))

    print("[Linear Regression (Normal)]: Max: " + str(max(accuracy["Linear_Normal"])))
    print("[Linear Regression (Normal)]: Min: " + str(min(accuracy["Linear_Normal"])))
    print("[Linear Regression (Normal)]: Mean: " + str(np.mean(accuracy["Linear_Normal"])))
    print("[Linear Regression (Normal)]: Standard Deviation: " + str(np.std(accuracy["Linear_Normal"])))

    print("[Linear Regression (Forced)]: Max: " + str(max(accuracy["Linear_Forced"])))
    print("[Linear Regression (Forced)]: Min: " + str(min(accuracy["Linear_Forced"])))
    print("[Linear Regression (Forced)]: Mean: " + str(np.mean(accuracy["Linear_Forced"])))
    print("[Linear Regression (Forced)]: Standard Deviation: " + str(np.std(accuracy["Linear_Forced"])))

    print("[Lasso]: Max: " + str(max(accuracy["Lasso"])))
    print("[Lasso]: Min: " + str(min(accuracy["Lasso"])))
    print("[Lasso]: Mean: " + str(np.mean(accuracy["Lasso"])))
    print("[Lasso]: Standard Deviation: " + str(np.std(accuracy["Lasso"])))

    print("[Ridge]: Max: " + str(max(accuracy["Ridge"])))
    print("[Ridge]: Min: " + str(min(accuracy["Ridge"])))
    print("[Ridge]: Mean: " + str(np.mean(accuracy["Ridge"])))
    print("[Ridge]: Standard Deviation: " + str(np.std(accuracy["Ridge"])))
