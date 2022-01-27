from Exercise3.DatasetGeneration import generate_alternative_dataset
from Exercise4.Exercise_4 import *
from Exercise3.ApproachTesting import check_truthfulness


def setup_tests_():

    ## TEST 1: sigma=0.0001
    dataset = generate_dataset(100000, sigma=0.0001)
    run_test(dataset)
    print("Dataset length: 100000; Sigma: e10-4 (almost exact star assignment);")

    ## TEST 2: sigma=0.3
    dataset = generate_dataset(100000, sigma=0.3)
    run_test(dataset)
    print("Dataset length: 100000; Sigma: 0.3; Truthfulness:")

    ## TEST 3: sigma=10
    dataset = generate_dataset(100000, sigma=10)
    run_test(dataset)
    print("Dataset length: 100000; Sigma: 10 (almost random star assignment);")

    ## TEST 4: Alternative approach
    dataset = generate_alternative_dataset(100000, 0.05)
    run_test(dataset)
    print("Dataset length: 100000; Probability of random increase/decrease: 5%;")


def run_test(data):
    accuracy = {}
    partitions = {}

    # Converts dataset to Pandas Dataframe
    df = pd.DataFrame(data, columns=['Food', 'Service', 'Value', 'Stars'])
    # Splits to create training and test set
    x, X, y, Y = load_train_test(80, df.values.tolist())

    log_normal = logistic_regression(x, X, y, Y)
    log_forced = logistic_regression_forced(x, X, y, Y)
    lin_normal = linear_regression(x, X, y, Y)
    lin_forced = linear_regression_forced(x, X, y, Y)
    lasso = lasso_regression(x, X, y, Y)
    ridge = ridge_regression(x, X, y, Y)

    accuracy["Logistic_Normal"] = compute_accuracy(data, log_normal)
    accuracy["Logistic_Forced"] = compute_accuracy(data, log_forced)
    accuracy["Linear_Normal"] = compute_accuracy(data, lin_normal)
    accuracy["Linear_Forced"] = compute_accuracy(data, lin_forced)
    accuracy["Lasso"] = compute_accuracy(data, lasso)
    accuracy["Ridge"] = compute_accuracy(data, ridge)

    partitions["Logistic_Normal"] = get_partitions(log_normal)
    partitions["Logistic_Forced"] = get_partitions(log_forced)
    partitions["Linear_Normal"] = get_partitions(lin_normal)
    partitions["Linear_Forced"] = get_partitions(lin_forced)
    partitions["Lasso"] = get_partitions(lasso)
    partitions["Ridge"] = get_partitions(ridge)

    # Print results:
    print("Analysis of accuracies: ")
    print("[Logistic Regression (Normal)]: Mean: " + str(accuracy["Logistic_Normal"]))
    print("Truthfulness?:", check_truthfulness(partitions["Logistic_Normal"][0], partitions["Logistic_Normal"][1], partitions["Logistic_Normal"][2]))

    print("[Logistic Regression (Forced)]: Mean: " + str(accuracy["Logistic_Forced"]))
    print("Truthfulness?:", check_truthfulness(partitions["Logistic_Forced"][0], partitions["Logistic_Forced"][1], partitions["Logistic_Forced"][2]))

    print("[Linear Regression (Normal)]: Mean: " + str(accuracy["Linear_Normal"]))
    print("Truthfulness?:", check_truthfulness(partitions["Linear_Normal"][0], partitions["Linear_Normal"][1], partitions["Linear_Normal"][2]))

    print("[Linear Regression (Forced)]: Mean: " + str(accuracy["Linear_Forced"]))
    print("Truthfulness?:", check_truthfulness(partitions["Linear_Forced"][0], partitions["Linear_Forced"][1], partitions["Linear_Forced"][2]))

    print("[Lasso]: Mean: " + str(accuracy["Lasso"]))
    print("Truthfulness?:", check_truthfulness(partitions["Lasso"][0], partitions["Lasso"][1], partitions["Lasso"][2]))

    print("[Ridge]: Mean: " + str(accuracy["Ridge"]))
    print("Truthfulness?:", check_truthfulness(partitions["Ridge"][0], partitions["Ridge"][1], partitions["Ridge"][2]))
