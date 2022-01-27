from joblib import Parallel, delayed
from ProfCode.priorityq import PriorityQueue
import networkx
import itertools as it
import math
from Common.Chunks import chunks
import numpy as np


def iterate(G, samples):
    if samples is None:
        samples = G.nodes()

    result = {}
    for u in samples:
        for v in G.nodes():
            if u != v:
                if (u, v) in G.edges() or (v, u) in G.edges():
                    result[frozenset([frozenset(np.array([u])), frozenset(np.array([v]))])] = 0
                else:
                    result[frozenset([frozenset(np.array([u])), frozenset(np.array([v]))])] = 1
    return result


def hierarchical_parallel(G, numJobs=1, numClusters=4):
    '''with Parallel(n_jobs=j) as parallel:
    Run in parallel diameter function on each processor by passing to each
    processor only the subset of nodes on which it works result = parallel(delayed(diameter)(graph, X) for X in chunks(
    graph.nodes(), math.ceil(len(graph.nodes()) / j)))
    Aggregates the results for res in result: if res > diam: diam = res return diam '''
    # Create a priority queue with each pair of nodes indexed by distance

    pq = PriorityQueue()

    results = []
    with Parallel(n_jobs=numJobs) as parallel:
        results = parallel(delayed(iterate)(G, X) for X in chunks(G.nodes(), math.ceil(len(G.nodes()) / numJobs)))
    for res in results:
        for key in res:
            pq.add(key, res[key])

    # Start with a cluster for each node
    clusters = set()
    for node in G.nodes():
        clusters.add(frozenset(np.array([node])))

    # l = list(pq.pq)
    # for elem in l:
    #     print(elem)

    while len(clusters) > numClusters:
        # Merge closest clusters
        s = list(pq.pop())
        clusters.remove(s[0])
        clusters.remove(s[1])

        # Update the distance of other clusters from the merged cluster
        for w in clusters:
            e1 = pq.remove(frozenset(np.array([s[0], w])))
            e2 = pq.remove(frozenset(np.array([s[1], w])))
            if e1 == 0 or e2 == 0:
                pq.add(frozenset(np.array([s[0] | s[1], w])), 0)
            else:
                pq.add(frozenset(np.array([s[0] | s[1], w])), 1)

        clusters.add(s[0] | s[1])

    return clusters
