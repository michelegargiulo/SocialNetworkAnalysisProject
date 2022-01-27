import time
import math
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from Common.DynamicGraphGeneration import get_random_graph
from Common.FixedGraph import get_fixed_graph
from Exercise2.Betweenness import betweenness
from Exercise2.Degree import degree_parallel
from Exercise2.Closeness import closeness_parallel
from Exercise2.HITS import hits_parallel
from Exercise2.NaiveImplementations import degree, closeness, page_rank, hits
from Exercise1.Betweenness import betweenness_parallel
from Exercise2.PageRank import page_rank_parallel


def degree_example_run():
    graph = get_fixed_graph()
    degmap = degree_parallel(graph, 4)
    return degmap


def degree_parallel_benchmark(numNodes=100, densities=[100], jobs=8):
    # map = degree(graph)
    # start = time.prin
    # naive = get_top_degree(map, small_graph.number_of_nodes())
    # print(naive)
    # optimized = degree_parallel(small_graph, small_graph.number_of_nodes(), 8)

    # Code for benchmarking, comparison and performance evaluation
    sum_optimized = 0
    sum_naive = 0
    optimized = []
    naive = []
    for j in range(len(densities)):
        # Max number of edges for N nodes is N*(N-1)/2. Density is a factory which goes from 0 to 1
        graph_dynamic = get_random_graph(numNodes, math.ceil((numNodes * (numNodes - 1)) * 0.5 * densities[j]))

        start = time.perf_counter()
        _ = degree_parallel(graph_dynamic, jobs)
        stop = time.perf_counter()
        sum_optimized += ((stop - start) / 100)
        optimized.append(stop - start)

        start = time.perf_counter()
        _ = degree(graph_dynamic)
        stop = time.perf_counter()
        sum_naive += ((stop - start) / 100)
        naive.append(stop - start)

    print("[Parallel Degree] Average elapsed time for naive algorithm: " + str(sum_naive))
    print("[Parallel Degree] Average elapsed time for optimized algorithm: " + str(sum_optimized))
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
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.show()


def closeness_example_run():
    graph = get_fixed_graph()
    result = closeness_parallel(graph, 4)
    return result


def closeness_parallel_benchmark(numNodes=100, densities=[100], jobs=8):
    # Code for benchmarking, comparison and performance evaluation
    sum_optimized = 0
    sum_naive = 0
    optimized = []
    naive = []
    for j in range(len(densities)):
        # Max number of edges for N nodes is N*(N-1)/2. Density is a factory which goes from 0 to 1
        graph_dynamic = get_random_graph(numNodes, math.ceil((numNodes * (numNodes - 1)) * 0.5 * densities[j]))

        start = time.perf_counter()
        _ = closeness_parallel(graph_dynamic, jobs)
        stop = time.perf_counter()
        sum_optimized += ((stop - start) / 100)
        optimized.append(stop - start)

        start = time.perf_counter()
        _ = closeness(graph_dynamic)
        stop = time.perf_counter()
        sum_naive += ((stop - start) / 100)
        naive.append(stop - start)

    print("[Parallel Closeness] Average elapsed time for naive algorithm: " + str(sum_naive))
    print("[Parallel Closeness] Average elapsed time for optimized algorithm: " + str(sum_optimized))
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
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.show()


def betweenness_example_run():
    graph = get_fixed_graph()
    result = betweenness_parallel(graph, num_jobs=4)
    return result


def betweenness_parallel_benchmark(numNodes=100, densities=[100], jobs=8):
    # Code for benchmarking, comparison and performance evaluation
    sum_optimized = 0
    sum_naive = 0
    optimized = []
    naive = []
    for j in range(len(densities)):
        # Max number of edges for N nodes is N*(N-1)/2. Density is a factory which goes from 0 to 1
        graph_dynamic = get_random_graph(numNodes, math.ceil((numNodes * (numNodes - 1)) * 0.5 * densities[j]))

        start = time.perf_counter()
        _ = betweenness_parallel(graph_dynamic, jobs)
        stop = time.perf_counter()
        sum_optimized += ((stop - start) / 100)
        optimized.append(stop - start)

        start = time.perf_counter()
        _ = betweenness(graph_dynamic)
        stop = time.perf_counter()
        sum_naive += ((stop - start) / 100)
        naive.append(stop - start)

    print("[Parallel Betweenness] Average elapsed time for naive algorithm: " + str(sum_naive))
    print("[Parallel Betweenness] Average elapsed time for optimized algorithm: " + str(sum_optimized))
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
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.show()


def page_rank_example_run():
    graph = get_fixed_graph()
    result = page_rank(graph)
    return result


def page_rank_parallel_benchmark(numNodes=100, densities=[100], jobs=8):
    # Code for benchmarking, comparison and performance evaluation
    sum_optimized = 0
    sum_naive = 0
    optimized = []
    naive = []
    for j in range(len(densities)):
        # Max number of edges for N nodes is N*(N-1)/2. Density is a factory which goes from 0 to 1
        graph_dynamic = get_random_graph(numNodes, math.ceil((numNodes * (numNodes - 1)) * 0.5 * densities[j]))

        start = time.perf_counter()
        _ = page_rank_parallel(graph_dynamic, num_jobs=jobs)
        stop = time.perf_counter()
        sum_optimized += ((stop - start) / 100)
        optimized.append(stop - start)

        start = time.perf_counter()
        _ = page_rank(graph_dynamic)
        stop = time.perf_counter()
        sum_naive += ((stop - start) / 100)
        naive.append(stop - start)

    print("[Parallel Page Rank] Average elapsed time for naive algorithm: " + str(sum_naive))
    print("[Parallel Page Rank] Average elapsed time for optimized algorithm: " + str(sum_optimized))
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
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.show()


def hits_parallel_benchmark(numNodes=100, densities=[100], jobs=8):
    # Code for benchmarking, comparison and performance evaluation
    sum_optimized = 0
    sum_naive = 0
    optimized = []
    naive = []
    for j in range(len(densities)):
        # Max number of edges for N nodes is N*(N-1)/2. Density is a factory which goes from 0 to 1
        graph_dynamic = get_random_graph(numNodes, math.ceil((numNodes * (numNodes - 1)) * 0.5 * densities[j]))

        start = time.perf_counter()
        _ = hits_parallel(graph_dynamic, num_jobs=jobs)
        stop = time.perf_counter()
        sum_optimized += ((stop - start) / 100)
        optimized.append(stop - start)

        start = time.perf_counter()
        _ = hits(graph_dynamic)
        stop = time.perf_counter()
        sum_naive += ((stop - start) / 100)
        naive.append(stop - start)

    print("[Parallel Hits] Average elapsed time for naive algorithm: " + str(sum_naive))
    print("[Parallel Hits] Average elapsed time for optimized algorithm: " + str(sum_optimized))
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
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.show()


def page_rank_tuning():
    pass


def hits_example_run():
    graph = get_fixed_graph()
    result = hits(graph)
    return result


def hits_tuning(max_iter=None, tolerances=None):
    if max_iter is None:
        max_iter = [50, 100, 200, 500, 1000]
    if tolerances is None:
        tolerances = list(np.linspace(10e-8, 10e-2, 1000))

    results = {}
    for iter in tqdm(max_iter):
        for tolerance in tqdm(tolerances, leave=False):
            graph = get_random_graph(100, math.ceil(4950 * 0.25))  # (n*(n-1)/2) * 0.25
            results[(iter, tolerance)] = hits(graph, int(iter), tolerance)

    print(results)