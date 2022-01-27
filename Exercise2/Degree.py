import networkx as nx
import itertools as it
from joblib import Parallel, delayed
from ProfCode.priorityq import PriorityQueue
import math
from Common.Chunks import chunks
from Common.Top import get_top_nodes

# with Parallel(n_jobs=numJobs) as parallel:
#     result = parallel(delayed(iterate)(graph, X, u) for X in chunks(graph.nodes(), math.ceil(len(graph.nodes()) / numJobs)))


def degree(G, slice = None):
    if slice is None:
        slice = G.nodes()

    cen = dict()
    for u in slice:
        cen[u] = nx.degree(G, u)
    return cen


def degree_parallel(graph, numJobs = 4):
    total = dict()
    result = []
    num_nodes_per_chunk = math.ceil(len(graph.nodes()) / numJobs)
    with Parallel(n_jobs=numJobs) as parallel:
        result = parallel(delayed(degree)(graph, X) for X in chunks(graph.nodes(), num_nodes_per_chunk))
    for res in result:
        total = {**total, **res}

    return total
