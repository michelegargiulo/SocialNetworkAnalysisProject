import os
import sys

import networkx as nx

from Common.DynamicGraphGeneration import get_random_graph
from Common.DrawGraph import DrawGraph
from Common.FixedGraph import get_fixed_graph
from Common.LoadGraph import load_graph
from Exercise2.Degree import degree


def shapley_degree(graph, C=None):
    """
    Shapley value for the characteristic function
        value(C) = |C| + |N(C)|,
    where N(C) is the set of nodes outside C with at least one neighbor in C

    :param graph: Unweighted Networkx graph
    :param C: A coalition of players, it should be iterable
    :return: Shapley value for the characteristic function for all nodes of the coalition
    """

    if C is None:
        return 0

    # If the graph is the graph is directed, the degree of a node is its in_degree
    if not nx.is_directed(graph):
        deg = degree(graph)
    else:
        deg = {i: graph.in_degree(i) for i in graph.nodes()}

    # Shapley values dictionary
    shapley = {}

    for v in C:
        shapley[v] = 1 / (1 + deg[v])
        for u in graph.neigbors(v):
            shapley[v] += 1 / (1 + deg[u])

    return shapley


def shapley_threshold(graph, k, C=None):
    """
    Shapley value for the characteristic function
        value(C) = |C| + |N(C,k)|,
    where N(C,k) is the set of nodes outside C with at least k neighbors in C

    :param graph: Unweighted Networkx graph
    :param C: A coalition of players, it should be iterable
    :param k: Threshold
    :return: Shapley value for the characteristic function for all nodes of the coalition
    """

    if C is None:
        return 0

    # If the graph is the graph is directed, the degree of a node is its in_degree
    if not nx.is_directed(graph):
        deg = degree(graph)
    else:
        deg = {i: graph.in_degree(i) for i in graph.nodes()}

    # Shapley values dictionary
    shapley = {}

    for v in C:
        shapley[v] = min(1, (k/(1+deg[v])))
        for u in graph.neighbors(v):
            shapley[v] += max(0, ((deg[u] - k + 1)/(deg[u] * (1 + deg[u]))))

    return shapley


# Il grafo deve essere pesato? bisogna usare una funzione della distanza come nel paper o solo distanza?
def shapley_closeness(graph):
    """

    :param graph:
    :param C: A coalition of players, it should be iterable
    :return:
    """
    shapley = {}

    for v in graph.nodes():
        shapley[v] = 0

    test = 0
    for v in graph.nodes():


        test += 1
        nodes, distances = custom_BFS_full(graph, v)
        index = len(nodes) - 1
        sum = 0
        prevDistance = -1
        prevSV = -1

        while index > 0:
            if distances[index] == prevDistance:
                currSV = prevSV
            else:
                currSV = (f_func(distances[index])/(1+index)) - sum

            shapley[nodes[index]] += currSV
            sum += f_func(distances[index])/(index*(1+index))
            prevDistance = distances[index]
            prevSV = currSV
            index -= 1
        shapley[v] += f_func(0) - sum

    return shapley

def custom_BFS(graph, C):
    level = 1
    n = len(graph.nodes())

    clevel = [u for u in C]
    visited = set(C)
    dist = {u : 0 for u in C}

    while len(visited) < n:
        nlevel = []
        while len(clevel) > 0:
            c = clevel.pop()
            for v in graph[c]:
                if v not in visited:
                    visited.add(v)
                    nlevel.append(v)
                    dist[v] = level
        level += 1
        clevel = nlevel

    return dist

def custom_BFS_full(graph, u):
    level = 1
    n = graph.number_of_nodes()
    clevel = [u]
    visited = []
    visited.append(u)
    dist = {}

    while len(visited) < n:
        nlevel = []
        if len(clevel) == 0 and level == 100:
            sys.exit()
        while len(clevel) > 0:
            c = clevel.pop()
            for v in graph[c]:
                if v not in visited:
                    visited.append(v)
                    nlevel.append(v)
                    dist[v] = level
        level += 1
        clevel = nlevel

    return list(dist.keys()), list(dist.values())


def f_func(dist):
    return 1/(1+dist)

# num_nodes = 0
# #while num_nodes != 20:
# #    graph = get_random_graph(20, 30, True)
# #    num_nodes = graph.number_of_nodes()
#
# dirname = "C:/Users/psabi/PycharmProjects/examProject"
# small_graph_path = os.path.join(dirname, "Dataset/musae_facebook_edges.csv")
# graph = load_graph(small_graph_path)
# #DrawGraph(graph)
# shapley = shapley_closeness(graph)
# print(shapley)

