from joblib import Parallel, delayed
from networkx import convert_node_labels_to_integers

from Common.Chunks import chunks
import networkx as nx
import numpy as np
import math
from Common.Normalize import normalize


def naive_hits(graph, max_iterations=100, tolerance=1.0e-8):
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


def _hubs_update(graph, nodes=None, adj_matrix=None, h=None, a=None):
    if adj_matrix is None:
        adj_matrix = nx.adjacency_matrix(graph, nodelist=nodes).todense()

    nodes = [int(i) for i in nodes]
    h[nodes] = adj_matrix[nodes, :].dot(a)


def _authorities_update(graph, nodes=None, adj_matrix_T=None, h=None, a=None):
    if adj_matrix_T is None:
        adj_matrix_T = nx.adjacency_matrix(graph, nodelist=nodes).T.todense()

    nodes = [int(i) for i in nodes]
    a[nodes] = adj_matrix_T[nodes, :].dot(h)


def hits_parallel(graph, max_iterations=100, tolerance=1.0e-8, num_jobs=4):
    """
    Parallel implementation of HITS algorithm
    :param graph: Networkx graph
    :param max_iterations: Maximum number of iterations
    :param tolerance: Tolerance for error
    :param num_jobs: Number of jobs
    :return: Hubs and authorities normalized dictionaries if algorithm converges, -1 otherwise.
    """

    # Check if the graph is directed
    if not nx.is_directed(graph):
        graph = graph.to_directed()

    node_list = list(graph.nodes())

    # Get adjacency matrix of the graph
    adj_matrix = nx.adjacency_matrix(graph, nodelist=node_list)

    # Get Transposed adjacency matrix
    adj_matrix_T = adj_matrix.T

    number_of_nodes = graph.number_of_nodes()

    # hubs and authorities initializations
    h = np.full((number_of_nodes, 1), 1. / number_of_nodes)
    a = np.full((number_of_nodes, 1), 1. / number_of_nodes)

    with Parallel(n_jobs=num_jobs, require='sharedmem') as parallel:
        for i in range(0, max_iterations):
            # old values of hubs and authorities
            h_old = h.copy()
            a_old = a.copy()

            # Update hubs h = adj_M * a
            results = parallel(delayed(_hubs_update)(graph, nodes=X, adj_matrix=adj_matrix, h=h, a=a)
                               for X in chunks(graph.nodes, math.ceil(len(node_list) / num_jobs)))

            # Update authorities a = adj_M_T * h
            results = parallel(delayed(_authorities_update)(graph, nodes=X, adj_matrix_T=adj_matrix_T, h=h, a=a)
                               for X in chunks(graph.nodes(), math.ceil(len(node_list) / num_jobs)))

            # Normalizing authorities and hubs
            a = normalize(a)
            h = normalize(h)

            # Get L1 norm for both hubs and authorities
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
