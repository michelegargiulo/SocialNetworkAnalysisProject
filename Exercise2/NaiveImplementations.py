from ProfCode.priorityq import PriorityQueue
import numpy as np
import networkx as nx
from Common.Normalize import normalize


# The measure associated to each node is exactly its degree
def degree(G):
    cen = dict()
    for u in G.nodes():
        cen[u] = G.degree(u)
    return cen


# The measure associated to each node is the sum of the (shortest) distances of this node from each remaining node
def closeness(G):
    cen = dict()

    for u in G.nodes():
        visited = set()
        visited.add(u)
        queue = [u]
        dist = dict()
        dist[u] = 0

        while len(queue) > 0:
            v = queue.pop(0)
            for w in G[v]:
                if w not in visited:
                    visited.add(w)
                    queue.append(w)
                    dist[w] = dist[v] + 1

        cen[u] = sum(dist.values())

    return cen


def page_rank(graph, beta=0.85, max_iterations=100, tolerance=1.0e-6):
    """
    Naive implementation of PageRank algorithm.
    :param graph: Networkx graph.
    :param beta:
    :param max_iterations: Maximum number of iterations.
    :param tolerance: Tolerance for error.
    :return: PageRank dict if algorithm converges, -1 otherwise.
    """

    # Check if the graph is directed
    if not nx.is_directed(graph):
        graph = graph.to_directed()

    number_of_nodes = graph.number_of_nodes()
    node_list = list(graph.nodes())

    # Get adjacency matrix of the graph
    adj_matrix = nx.adjacency_matrix(graph, nodelist=node_list)

    # Get out-degree from adjacency matrix
    out_degree = adj_matrix.sum(axis=0)

    # Calculating Transition Matrix
    M = adj_matrix.multiply(1. / out_degree)

    v = np.full((number_of_nodes, 1), 1. / number_of_nodes)

    for i in range(0, max_iterations):
        v_old = v
        v = beta * M.dot(v) + (1 - beta) / number_of_nodes

        error = np.absolute(v - v_old).sum()
        if error < number_of_nodes * tolerance:
            # Insert solution in a dictionary with nodes as keys and PageRank as value
            page_rank_dict = {}
            for node, el in zip(node_list, range(0, number_of_nodes)):
                page_rank_dict[node] = v[el][0]
            return page_rank_dict
    print("PageRank algorithm does not converge!")
    return -1


def hits(graph, max_iterations=100, tolerance=1.0e-8):
    """
    Naive implementation of HITS algorithm
    :param graph: Networkx graph
    :param max_iterations: Maximum number of iterations
    :param tolerance: Tolerance for error
    :return: Hubs and authorities normalized dictionaries if algorithm converges, -1 otherwise.
    """

    # Check if the graph is directed
    if not nx.is_directed(graph):
        graph = graph.to_directed()

    number_of_nodes = graph.number_of_nodes()
    node_list = list(graph.nodes())

    # Get adjacency matrix of the graph
    adj_matrix = nx.adjacency_matrix(graph, nodelist=node_list)

    # Get Transposed adjacency matrix
    adj_matrix_T = adj_matrix.T

    # hubs and authorities initializations
    h = np.full((number_of_nodes, 1), 1. / number_of_nodes)
    a = np.full((number_of_nodes, 1), 1. / number_of_nodes)

    for i in range(0, max_iterations):
        # old value of hubs and authorities
        h_old = h
        a_old = a

        # Update hubs h = adj_M * a
        h = adj_matrix.dot(a)

        # Update authorities a = adj_M_T * h
        a = adj_matrix_T.dot(h)

        # Normalizing authorities and hubs
        a = normalize(a)
        h = normalize(h)

        # Error for both hubs and authorities
        error_h = np.absolute(h - h_old).sum()
        error_a = np.absolute(a - a_old).sum()

        # Check if errors are less than tolerance
        if error_h < tolerance and error_a < tolerance:
            # Insert solution in two dictionaries with nodes as keys and hub/authority as value
            h_dict = {}
            a_dict = {}

            h_sum = h.sum()
            a_sum = a.sum()

            for node, el in zip(node_list, range(0, number_of_nodes)):
                h_dict[node] = h[el][0] / h_sum
                a_dict[node] = a[el][0] / a_sum
            return h_dict, a_dict
    print("HITS algorithm does not converge!")
    return -1
