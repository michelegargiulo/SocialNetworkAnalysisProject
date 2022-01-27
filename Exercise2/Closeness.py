import networkx as nx
import itertools as it
from Common.Chunks import chunks
from joblib import Parallel, delayed
import math


def iterate(graph, sample):

    cen = {}

    if sample is None:
        sample = graph.nodes()

    for u in sample:
        visited = set()
        visited.add(u)
        queue = [u]
        dist = dict()
        dist[u] = 0

        while len(queue) > 0:
            v = queue.pop(0)
            for w in graph[v]:
                if w not in visited:
                    visited.add(w)
                    queue.append(w)
                    dist[w] = dist[v] + 1
        cen[u] = (len(dist.values())-1)/sum(dist.values())
    return cen


def closeness_parallel(graph, numJobs = 4):
    cen = dict()

    results = []
    with Parallel(n_jobs=numJobs) as parallel:
        results = parallel(delayed(iterate)(graph, X) for X in chunks(graph.nodes(), math.ceil(len(graph.nodes()) / numJobs)))

    for res in results:
        cen = {**cen, **res}
    return cen
