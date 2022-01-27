import random

from Common.ClusterSimilarity import compare_results
from Common.DrawGraph import DrawGraph
from Common.DynamicGraphGeneration import remove_small_components, get_biggest_subgraph
from Common.LoadGraph import load_graph
from Common.FixedGraph import get_fixed_graph
from Common.TargetCluster import build_target_sets
from Common.Top import get_top_nodes

from Exercise1.Betweenness import girvan_newman_parallel
from Exercise1.Hierarchical import hierarchical_parallel
from Exercise1.NaiveImplementations import hierarchical
from Exercise1.PerformanceTest import hierarchical_parallel_example_run, girvan_newman_example_run, \
    girvan_newman_parallel_benchmark, hierarchical_parallel_benchmark, kmeans_optimized_example_run, \
    kmeans_optimized_benchmark, spectral_example_run
from Exercise1.KMeans import four_means_optimized
from Exercise1.Spectral import spectral_four_clusters
from Exercise2.Betweenness import betweenness_parallel
from Exercise2.PageRank import page_rank_parallel

from Exercise2.PerformanceTest import degree_parallel_benchmark, closeness_example_run, closeness_parallel_benchmark, \
    betweenness_example_run, degree_example_run, page_rank_example_run, hits_example_run, hits_tuning, \
    betweenness_parallel_benchmark, hits_parallel_benchmark, page_rank_parallel_benchmark
from Exercise2.NaiveImplementations import degree, page_rank, hits
from Exercise2.Degree import degree_parallel
from Exercise2.Closeness import closeness_parallel

from Exercise3.DatasetGeneration import read_dataset
from Exercise3.DatasetGeneration import get_probabilities
from Exercise3.GraphGeneration import create_graph
from Exercise3.ApproachTesting import setup_tests

from Exercise4.Exercise_4 import run_exercise_4
from Exercise4.ApproachTesting import setup_tests_

import os
import numpy as np
import time
import pickle
import networkx as nx


# loading graphs
dirname = os.path.dirname(__file__)
full_graph_path = os.path.join(dirname, 'Dataset/musae_facebook_edges.csv')

example_graph = get_fixed_graph()
full_graph = load_graph(full_graph_path)

half_graph = nx.subgraph(full_graph, list(full_graph.nodes())[:10000])
huge_graph = nx.subgraph(full_graph, list(full_graph.nodes())[:5000])
large_graph = nx.subgraph(full_graph, list(full_graph.nodes())[:2500])
medium_graph = nx.subgraph(full_graph, list(full_graph.nodes())[:1000])
small_graph = nx.subgraph(full_graph, list(full_graph.nodes())[:500])
tiny_graph = nx.subgraph(full_graph, list(full_graph.nodes())[:200])
# half_graph = get_random_nodes(full_graph, 10000)
# huge_graph = get_random_nodes(full_graph, 5000)
# large_graph = get_random_nodes(full_graph, 2500)
# medium_graph = get_random_nodes(full_graph, 1000)
# small_graph = get_random_nodes(full_graph, 500)
# tiny_graph = get_random_nodes(full_graph, 200)
restaurants_dataset = read_dataset()

# Remove non-maximal isolated connected components, since the algorithms require that the graphs are connected
half_graph = get_biggest_subgraph(half_graph.copy())
huge_graph = get_biggest_subgraph(huge_graph.copy())
large_graph = get_biggest_subgraph(large_graph.copy())
medium_graph = get_biggest_subgraph(medium_graph.copy())
small_graph = get_biggest_subgraph(small_graph.copy())
tiny_graph = get_biggest_subgraph(tiny_graph.copy())

RUN_EXAMPLE = False
RUN_BENCHMARKS = False
RUN_DATASET = True

# full_graph = small_graph
# huge_graph = small_graph
# large_graph = small_graph
# medium_graph = small_graph
# small_graph = small_graph


'''
Showcase the algorithms by running them on a fixed graph (which can be found in Common.FixedGraph file) 
and show the results
'''
RUN_EXAMPLE = input("Do you want to run the algorithms on a fixed small graph? Y=True, other=False: ").lower() == "y"
if RUN_EXAMPLE:

    # CLUSTERING (Exercise 1) ##########################################################################################

    ## Hierarchical
    if input("Do you want to execute an example run of the HIERARCHICAL CLUSTERING algorithm? y/n: ").lower() == 'y':
        print(hierarchical_parallel_example_run())

    ## KMeans
    if input("Do you want to execute an example run of the K-MEANS CLUSTERING algorithm? y/n: ").lower() == 'y':
        print(kmeans_optimized_example_run())

    ## Spectral
    if input("Do you want to execute an example run of the SPECTRAL CLUSTERING algorithm? y/n: ").lower() == 'y':
        print(spectral_example_run())

    ## Girvan-Newman
    if input("Do you want to execute an example run of the GIRVAN-NEWMAN CLUSTERING algorithm? y/n: ").lower() == 'y':
        print(girvan_newman_example_run())

    # CENTRALITY INDICES (Exercise 2) ##################################################################################

    ## Closeness
    if input("Do you want to execute an example run of the CLOSENESS algorithm? y/n: ").lower() == 'y':
        print(closeness_example_run())

    ## Betweenness
    if input("Do you want to execute an example run of the BETWEENNESS algorithm? y/n: ").lower() == 'y':
        print(betweenness_example_run())

    ## Degree
    if input("Do you want to execute an example run of the DEGREE algorithm? y/n: ").lower() == 'y':
        print(degree_example_run())

    ## Page Rank
    if input("Do you want to execute an example run of the PAGE RANK algorithm? y/n: ").lower() == 'y':
        print(page_rank_example_run())

    ## HITS
    if input("Do you want to execute an example run of the HITS algorithm? y/n: ").lower() == 'y':
        print(hits_example_run())

'''
Benchmarks the optimized algorithms comparing them with the naive implementations
and plot the results
'''
RUN_BENCHMARKS = input("Do you want to run benchmarks? 100 random graphs with increasing edge density will be "
                       "generated Y=True, other=False: ").lower() == "y"
if RUN_BENCHMARKS:

    # CLUSTERING (Exercise 1) ##########################################################################################

    ## Hierarchical
    if input("Do you want to run a benchmark for the HIERARCHICAL CLUSTERING (PARALLEL) algorithm? y/n: ").lower() == 'y':
        hierarchical_parallel_benchmark(100, list(np.linspace(0.1, 0.4, 100)), 4, 4)
        print("[Hierarchical Benchmark] Done!")

    ## KMeans
    if input("Do you want to run a benchmark for the K-MEANS CLUSTERING (OPTIMIZED) algorithm? y/n: ").lower() == 'y':
        kmeans_optimized_benchmark(100, list(np.linspace(0.1, 0.4, 100)))
        print("[KMeans Benchmark] Done!")

    ## Girvan-Newman
    if input("Do you want to run a benchmark for the GIRVAN-NEWMAN CLUSTERING (PARALLEL) algorithm? y/n: ").lower() == 'y':
        girvan_newman_parallel_benchmark(100, list(np.linspace(0.05, 0.15, 100)), numClusters=4, numJobs=4)
        print("[Girvan-Newman Benchmark] Done!")

    # CENTRALITY INDICES (Exercise 2) ##################################################################################

    ## Degree
    if input("Do you want to run a benchmark for the DEGREE (PARALLEL) algorithm? y/n: ").lower() == 'y':
        degree_parallel_benchmark(100, list(np.linspace(0.1, 0.4, 100)), 8)
        print("[Degree Benchmark] Done!")

    ## Closeness
    if input("Do you want to run a benchmark for the CLOSENESS (PARALLEL) algorithm? y/n: ").lower() == 'y':
        closeness_parallel_benchmark(100, list(np.linspace(0.1, 0.4, 100)), 4)
        print("[Closeness Benchmark] Done!")

    ## Betweenness
    if input("Do you want to run a benchmark for the BETWEENNESS (PARALLEL) algorithm? y/n: ").lower() == 'y':
        betweenness_parallel_benchmark(100, list(np.linspace(0.1, 0.4, 100)), 4)
        print("[Closeness Benchmark] Done!")

    ## Page Rank
    if input("Do you want to run a benchmark for the PAGE RANK (PARALLEL) algorithm? y/n: ").lower() == 'y':
        page_rank_parallel_benchmark(100, list(np.linspace(0.1, 0.4, 100)), 4)
        print("[Page Rank Benchmark] Done!")

    ## Hits
    if input("Do you want to run a benchmark for the HITS (PARALLEL) algorithm? y/n: ").lower() == 'y':
        hits_parallel_benchmark(100, list(np.linspace(0.1, 0.4, 100)), 4)
        print("[Hits Benchmark] Done!")


'''
Run the algorithm on the full dataset and get the results
'''
RUN_DATASET = input("Do you want to run the algorithms on the full dataset? Y=True, other=False: ").lower() == 'y'
if RUN_DATASET:

    # CLUSTERING (Exercise 1) ##########################################################################################

    # Hierarchical
    if input("Do you want to compute HIERARCHICAL CLUSTERING (PARALLEL) on the dataset? y/n: ").lower() == 'y':
        start = time.perf_counter()
        clusters = hierarchical_parallel(huge_graph, numJobs=2)
        stop = time.perf_counter()
        with open("hierarchical_clustering_parallel_results.txt", "wb") as f:
            pickle.dump(clusters, f)
        print("Elapsed time: " + str(stop - start))  # Takes around 45 seconds on 5000 nodes, runs out of memory for larger graphs
        print("Hierarchical clusters: ", clusters)
        print("Average similarity between clusters:", compare_results(build_target_sets(medium_graph), clusters)[0])

    # Girvan-Newman
    if input("Do you want to compute GIRVAN-NEWMAN CLUSTERING (PARALLEL) on the dataset? y/n: ").lower() == 'y':
        start = time.perf_counter()
        clusters = girvan_newman_parallel(huge_graph, numJobs=12)
        stop = time.perf_counter()
        with open("girvan_newman_parallel_results.txt", "wb") as f:
            pickle.dump(clusters, f)
        print("Elapsed time: " + str(stop - start))  # Takes around 162 seconds on 1000 nodes
        print("Girvan-Newman clusters:", clusters)
        print("Average similarity between clusters:", compare_results(build_target_sets(large_graph), clusters)[0])

    # KMeans
    if input("Do you want to compute K-MEANS CLUSTERING (OPTIMIZED) on the dataset? y/n: ").lower() == 'y':
        start = time.perf_counter()
        clusters = four_means_optimized(full_graph)
        stop = time.perf_counter()
        with open("kmeans_clustering_parallel_results.txt", "wb") as f:
            pickle.dump(clusters, f)
        print("Elapsed time: " + str(stop - start))
        print("K-Means clusters:", clusters)
        print("Average similarity between clusters:", compare_results(build_target_sets(full_graph), clusters)[0])

    # Spectral
    if input("Do you want to compute SPECTRAL CLUSTERING (SAMPLED) on the dataset? y/n: ").lower() == 'y':
        start = time.perf_counter()
        clusters = spectral_four_clusters(huge_graph)
        stop = time.perf_counter()
        with open("spectral_clustering_parallel_results.txt", "wb") as f:
            pickle.dump(clusters, f)
        print("Elapsed time: " + str(stop - start))
        print("Spectral clusters:", clusters)
        print("Average similarity between clusters:", compare_results(build_target_sets(large_graph), clusters)[0])

    # Degree
    if input("Do you want to compute DEGREE (PARALLEL) on the dataset? y/n: ").lower() == 'y':
        start = time.perf_counter()
        result = degree_parallel(full_graph, 4)  # Takes around 8.5 seconds
        top = get_top_nodes(result, 500)
        stop = time.perf_counter()
        with open("top500_degree.txt", "wb") as f:
            pickle.dump(top, f)
        print("Top 500 nodes for DEGREE: ")
        print(top)
        print("Elapsed time " + str(stop - start))

    # Closeness
    if input("Do you want to compute CLOSENESS (PARALLEL) on the dataset? y/n: ").lower() == 'y':
        start = time.perf_counter()
        result = closeness_parallel(full_graph, 8)  # Takes around 303 seconds
        top = get_top_nodes(result, 500)
        stop = time.perf_counter()
        with open("top500_closeness.txt", "wb") as f:
            pickle.dump(top, f)
        print("Top 500 nodes for CLOSENESS: ")
        print(top)
        print("[Closeness] Elapsed time " + str(stop - start))
        # print(nx.closeness_centrality())

    # Betweenness
    if input("Do you want to compute BETWEENNESS (PARALLEL) on the dataset? y/n: ").lower() == 'y':
        start = time.perf_counter()
        _, result = betweenness_parallel(full_graph, 12)  # Takes around 1902 seconds
        print(result)
        with open("betw_result.txt", "wb") as f:
            pickle.dump(result, f)
        top = get_top_nodes(result, 500)
        stop = time.perf_counter()
        with open("top500_betweenness.txt", "wb") as f:
            pickle.dump(top, f)
        print("Top 500 nodes for BETWEENNESS: ")
        print(top)
        print("[Betweenness] Elapsed time " + str(stop - start))

    # Page Rank
    if input("Do you want to compute PAGE RANK (NAIVE) on the dataset? y/n: ").lower() == 'y':
        start = time.perf_counter()
        result = page_rank(full_graph)  # Takes around 2 seconds
        top = get_top_nodes(result, 500)
        stop = time.perf_counter()
        with open("top500_page_rank.txt", "wb") as f:
            pickle.dump(top, f)
        print("Top 500 nodes for PAGE RANK: ")
        print(top)
        print("Elapsed time " + str(stop - start))

    # HITS
    if input("Do you want to compute HITS (NAIVE) on the dataset? y/n: ").lower() == 'y':
        start = time.perf_counter()
        result_hubs, result_auth = hits(full_graph)  # Takes aound 2 seconds
        top_hubs = get_top_nodes(result_hubs, 500)
        top_auth = get_top_nodes(result_auth, 500)
        stop = time.perf_counter()
        with open("top500_hubs_hits.txt", "wb") as f:
            pickle.dump(top_hubs, f)
        with open("top500_auth_hits.txt", "wb") as f:
            pickle.dump(top_auth, f)
        print("Top 500 Hubs for HITS: ")
        print(top_hubs)
        print("Top 500 Auth for HITS: ")
        print(top_auth)
        print("Elapsed time " + str(stop - start))


'''
Run exercise 2
'''
RUN_EX2 = input("Do you want to run Exercise 2? Y=True, other=False: ").lower() == "y"
if RUN_EX2:
    print("This will run Page Rank and HITS on the dataset with different parameters")
    print("Page Rank: ")

    print("Hits: ")
    hits_tuning()



'''
Run exercise 3
'''
RUN_EX3 = input("Do you want to run Exercise 3? Y=True, other=False: ").lower() == "y"
if RUN_EX3:
    setup_tests()
    
    
'''
Run exercise 4
'''
RUN_EX4 = input("Do you want to run Exercise 4? Y=True, other=False: ").lower() == "y"
if RUN_EX4:
    # run_exercise_4()
    setup_tests_()


'''
Result Explorer
'''
EXPLORE_RESULTS = input("Do you want to print out the Cluster Similarity for the computed clusters? Y=True, other=False: ").lower() == "y"
if EXPLORE_RESULTS:
    # Cluster results
    clusters = set()

    # Hierarchical
    with open("hierarchical_clustering_parallel_results.txt", "rb") as f:
        clusters = pickle.load(f)
    print("Hierarchical clustering results [Huge Graph]: ", clusters, build_target_sets(medium_graph))

    # Betweenness
    with open("top500_betweenness.txt", "rb") as f:
        top = pickle.load(f)
    s = sum(top.values())
    for v in top:
        top[v] = top[v]/s
    print("Betweenness Top 500 nodes [Full Graph]: ", top)

    # Degree
    with open("top500_degree.txt", "rb") as f:
        top = pickle.load(f)
    print("Degree Top 500 nodes [Full Graph]: ", top)

    # Closeness
    with open("top500_closeness.txt", "rb") as f:
        top = pickle.load(f)
    s = sum(top.values())
    for v in top:
        top[v] = top[v]/s
    print("Closeness Top 500 nodes [Full Graph]: ", top)

    # Page Rank
    with open("top500_page_rank.txt", "rb") as f:
        top = pickle.load(f)
    print("Page Rank Top 500 nodes [Full Graph]: ", top)

    # Hits
    with open("top500_hubs_hits.txt", "rb") as f:
        top_h = pickle.load(f)
    with open("top500_auth_hits.txt", "rb") as f:
        top_a = pickle.load(f)
    print("Hits Top 500 nodes [Full Graph]:\nHubs:", top_h, "\nAuthorities:", top_a)


_ = input("Program terminated. Press any key to exit...")