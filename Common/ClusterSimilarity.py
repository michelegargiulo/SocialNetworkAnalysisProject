import random
import numpy as np


def compare_results(output1, output2):
    simil = []
    matrix = np.zeros((len(output1), len(output2)))

    for i, cluster1 in enumerate(output1):
        for j, cluster2 in enumerate(output2):
            matrix[i][j] = similarity(cluster1, cluster2)
    rows, cols = matrix.shape
    while rows > 0 and cols > 0:
        (i, j) = np.where(matrix == np.max(matrix))
        simil.append(matrix[i, j])
        matrix = np.delete(matrix, i, axis=0)  # Remove the row and the column of the maximum
        matrix = np.delete(matrix, j, axis=1)
        rows, cols = matrix.shape

    # Compute average similarity
    # Take into account the different number of clusters
    for _ in range(rows):
        simil.append(0)
    for _ in range(cols):
        simil.append(0)
    average = np.mean(np.array(simil))
    return average, simil


def similarity(cluster1, cluster2):
    num_total = len(set(cluster1).union(set(cluster2)))
    common = len(set(cluster1).intersection((set(cluster2))))
    return common / num_total if num_total > 0 else 0


def test_similarity():

    a = set()
    b = set()

    a = [1, 2, 3]
    b = [2, 3, 4]
    print(similarity(a, b))

    a = ["a", 2, 3]
    b = [2, 3, 4]
    print(similarity(a, b))

    a = ["a", "b", "c"]
    b = [2, 3, 4]
    print(similarity(a, b))

    a = [1, 2, 3, 4, 5]
    b = [1, 2, 3]
    print(similarity(a, b))

    a = [1, 2]
    b = [2]
    print(similarity(a, b))

    a = []
    b = [2, 3, 4]
    print(similarity(a, b))

    a = [1, 2, 3]
    b = [1, 2, 3]
    print(similarity(a, b))

    a = [1, "a"]
    b = ["a"]
    print(similarity(a, b))

    a = []
    b = []
    print(similarity(a, b))

    for _ in range(100):
        a = set()
        b = set()
        for x in range(15):
            if random.uniform(0, 100) < 50:
                a.add(x)
            if random.uniform(0, 100) < 50:
                b.add(x)

        print("Clusters and similarity:")
        print(a)
        print(b)
        print(similarity(a, b))