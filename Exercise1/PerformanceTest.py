import networkx as nx

from Exercise1.NaiveImplementations import four_means, hierarchical, girvan_newman
from Exercise1.Betweenness import girvan_newman_parallel
from Exercise1.KMeans import four_means_optimized
from Exercise1.Hierarchical import hierarchical_parallel
from Exercise1.Spectral import spectral_four_clusters
from Common.FixedGraph import get_fixed_graph
from Common.DynamicGraphGeneration import get_random_graph
from matplotlib import pyplot as plt
import time
import numpy as np
import math


def hierarchical_parallel_example_run(numJobs = 2):
    graph = get_fixed_graph()
    clusters = hierarchical_parallel(graph, numJobs)
    return clusters


def hierarchical_parallel_benchmark(numNodes, densities = [100], numJobs = 4, numClusters = 4):
    # Code for benchmarking, comparison and performance evaluation
    sum_optimized = 0
    sum_naive = 0
    optimized = []
    naive = []
    for j in range(len(densities)):
        # Max number of edges for N nodes is N*(N-1)/2. Density is a factory which goes from 0 to 1
        graph_dynamic = get_random_graph(numNodes, math.ceil((numNodes * (numNodes - 1)) * 0.5 * densities[j]))

        start = time.perf_counter()
        _ = hierarchical_parallel(graph_dynamic, numJobs, numClusters)
        stop = time.perf_counter()
        sum_optimized += ((stop - start) / 100)
        optimized.append(stop - start)

        start = time.perf_counter()
        _ = hierarchical(graph_dynamic, numClusters)
        stop = time.perf_counter()
        sum_naive += ((stop - start) / 100)
        naive.append(stop - start)

    print("[KMeans] Average elapsed time for naive algorithm: " + str(sum_naive))
    print("[KMeans] Average elapsed time for optimized algorithm: " + str(sum_optimized))
    labels = [x for x in range(100)]
    x = np.arange(len(labels))
    width = 2
    fig, ax = plt.subplots()
    if sum_naive < sum_optimized:
        ax.bar(x - width / 2, optimized, width, label="Optimized")
        ax.bar(x - width / 2, naive, width, label="Naive")
    else:
        ax.bar(x - width / 2, naive, width, label="Naive")
        ax.bar(x - width / 2, optimized, width, label="Optimized")
    ax.bar(x - width / 2, naive, width, label="Naive")
    ax.bar(x - width / 2, optimized, width, label="Optimized")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.show()


def kmeans_optimized_example_run():
    graph = get_fixed_graph()
    clusters = four_means_optimized(graph)
    return clusters


def kmeans_optimized_benchmark(numNodes, densities = [100]):
    # Code for benchmarking, comparison and performance evaluation
    sum_optimized = 0
    sum_naive = 0
    optimized = []
    naive = []
    for j in range(len(densities)):
        # Max number of edges for N nodes is N*(N-1)/2. Density is a factory which goes from 0 to 1
        graph_dynamic = get_random_graph(numNodes, math.ceil((numNodes * (numNodes - 1)) * 0.5 * densities[j]))

        start = time.perf_counter()
        _ = four_means_optimized(graph_dynamic)
        stop = time.perf_counter()
        sum_optimized += ((stop - start) / 100)
        optimized.append(stop - start)

        start = time.perf_counter()
        _ = four_means(graph_dynamic)
        stop = time.perf_counter()
        sum_naive += ((stop - start) / 100)
        naive.append(stop - start)

    print("[KMeans] Average elapsed time for naive algorithm: " + str(sum_naive))
    print("[KMeans] Average elapsed time for optimized algorithm: " + str(sum_optimized))
    labels = [x for x in range(100)]
    x = np.arange(len(labels))
    width = 2
    fig, ax = plt.subplots()
    if sum_naive < sum_optimized:
        ax.bar(x - width / 2, optimized, width, label="Optimized")
        ax.bar(x - width / 2, naive, width, label="Naive")
    else:
        ax.bar(x - width / 2, naive, width, label="Naive")
        ax.bar(x - width / 2, optimized, width, label="Optimized")
    ax.bar(x - width / 2, naive, width, label="Naive")
    ax.bar(x - width / 2, optimized, width, label="Optimized")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.show()


def spectral_example_run():
    graph = get_fixed_graph()
    # Remap the graph nodes to ints
    map = {}
    cnt = 0
    for node in graph.nodes():
        map[node] = cnt
        cnt += 1
    nx.relabel_nodes(graph, map, False)
    clusters = spectral_four_clusters(graph)
    return clusters


def girvan_newman_example_run():
    graph = get_fixed_graph()
    clusters = girvan_newman_parallel(graph, numClusters=4, numJobs=2)
    return clusters


def girvan_newman_parallel_benchmark(numNodes, densities = [100], numClusters=4, numJobs=4):
    # Code for benchmarking, comparison and performance evaluation
    sum_optimized = 0
    sum_naive = 0
    optimized = []
    naive = []
    for j in range(len(densities)):
        # Max number of edges for N nodes is N*(N-1)/2. Density is a factory which goes from 0 to 1
        graph_dynamic = get_random_graph(numNodes, math.ceil((numNodes * (numNodes - 1)) * 0.5 * densities[j]), True)

        start = time.perf_counter()
        _ = girvan_newman_parallel(graph_dynamic, numClusters=numClusters, numJobs=numJobs)
        stop = time.perf_counter()
        sum_optimized += ((stop - start) / 100)
        optimized.append(stop - start)

        start = time.perf_counter()
        _ = girvan_newman(graph_dynamic, numClusters=4)
        stop = time.perf_counter()
        sum_naive += ((stop - start) / 100)
        naive.append(stop - start)

    print("[Girvan-Newman] Average elapsed time for naive algorithm: " + str(sum_naive))
    print("[Girvan-Newman] Average elapsed time for optimized algorithm: " + str(sum_optimized))
    labels = [x for x in range(100)]
    x = np.arange(len(labels))
    width = 2
    fig, ax = plt.subplots()
    if sum_naive < sum_optimized:
        ax.bar(x - width / 2, optimized, width, label="Optimized")
        ax.bar(x - width / 2, naive, width, label="Naive")
    else:
        ax.bar(x - width / 2, naive, width, label="Naive")
        ax.bar(x - width / 2, optimized, width, label="Optimized")
    ax.bar(x - width / 2, naive, width, label="Naive")
    ax.bar(x - width / 2, optimized, width, label="Optimized")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.show()



'''
def hierarchical_parallel_benchmark(numNodes, densities = [100], numJobs=4):
    # Code for benchmarking, comparison and performance evaluation
    sum_optimized = 0
    sum_naive = 0
    optimized = []
    naive = []
    for j in range(len(densities)):
        # Max number of edges for N nodes is N*(N-1)/2. Density is a factory which goes from 0 to 1
        graph_dynamic = get_random_graph(numNodes, math.ceil((numNodes * (numNodes - 1)) * 0.5 * densities[j]))

        start = time.perf_counter()
        _ = hierarchical_parallel(graph_dynamic, numJobs)
        stop = time.perf_counter()
        sum_optimized += ((stop - start) / 100)
        optimized.append(stop - start)

        start = time.perf_counter()
        _ = hierarchical(graph_dynamic)
        stop = time.perf_counter()
        sum_naive += ((stop - start) / 100)
        naive.append(stop - start)

    print("[Hierarchical] Average elapsed time for naive algorithm: " + str(sum_naive))
    print("[Hierarchical] Average elapsed time for optimized algorithm: " + str(sum_optimized))
    labels = [x for x in range(100)]
    x = np.arange(len(labels))
    width = 2
    fig, ax = plt.subplots()
    if sum_naive < sum_optimized:
        ax.bar(x - width / 2, optimized, width, label="Optimized")
        ax.bar(x - width / 2, naive, width, label="Naive")
    else:
        ax.bar(x - width / 2, naive, width, label="Naive")
        ax.bar(x - width / 2, optimized, width, label="Optimized")
    ax.bar(x - width / 2, naive, width, label="Naive")
    ax.bar(x - width / 2, optimized, width, label="Optimized")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.show()
'''