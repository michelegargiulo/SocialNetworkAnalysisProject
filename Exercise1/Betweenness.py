from Exercise2.Betweenness import betweenness_parallel
from ProfCode.priorityq import PriorityQueue
from multiprocessing import Pool
import itertools
import networkx as nx
from Common.Chunks import chunks
from joblib import Parallel, delayed
import math


# Clusters are computed by iteratively removing edges of largest betweenness
def girvan_newman_parallel(graph, numClusters=4, numJobs=4):
    # eb,nb = betweenness_centrality_parallel(graph, numJobs)
    eb, nb = betweenness_parallel(graph, numJobs)
    # pq = PriorityQueue()
    # for i in eb.keys():
    #     pq.add(i, -eb[i])
    graph = graph.copy()

    while len(list(nx.connected_components(graph))) < numClusters:
        edge = tuple(max(eb, key=eb.get))
        graph.remove_edges_from([edge])
        eb, nb = betweenness_parallel(graph, numJobs)

    return list(nx.connected_components(graph))