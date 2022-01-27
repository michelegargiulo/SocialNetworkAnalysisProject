from joblib import Parallel, delayed
from Common.Chunks import chunks
import networkx as nx
import numpy as np
import math


def get_transition_matrix(graph, node_list):
    """
    Get Transition matrix of a graph
    :param graph: Networkx graph
    :param node_list: List of the nodes
    :return:
    """
    # Get adjacency matrix of the graph
    adj_matrix = nx.adjacency_matrix(graph, nodelist=node_list)

    # Get out-degree from adjacency matrix
    out_degree = adj_matrix.sum(axis=0)

    # Calculating Transition Matrix
    M = adj_matrix.multiply(1. / out_degree)

    return M


def naive_page_rank(graph, beta=0.85, max_iterations=100, tolerance=1.0e-6):
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
    M = get_transition_matrix(graph, node_list)

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


def _page_rank_update(graph, beta=0.85, transition_matrix=None, nodes=None, v=None):
    """
    PageRank update.
    :param graph: Networkx graph.
    :param beta:
    :param transition_matrix: Transition matrix for the provided graph
    :param nodes: List of nodes
    :param v: Page rank numpy array
    """

    number_of_nodes = graph.number_of_nodes()

    # Check if nodes is None
    if nodes is None:
        node_list = list(graph.nodes())
    else:
        node_list = list(nodes)

    # Check if a transition matrix is provided
    if transition_matrix is None:
        M = get_transition_matrix(graph, node_list).tocsr()
    else:
        M = transition_matrix

    # Check if PageRank numpy array is provided
    if v is None:
        v = np.full((number_of_nodes, 1), 1. / number_of_nodes)

    v_old = v.copy()
    node_list = [int(i) for i in node_list]

    # Update PageRank for the provided nodes
    v[node_list] = beta * M[node_list, :].dot(v_old) + (1 - beta) / number_of_nodes


def page_rank_parallel(graph, beta=0.85, max_iterations=100, tolerance=1.0e-6, num_jobs=4):
    """
    Parallel implementation of PageRank algorithm.
    :param graph: Networkx graph.
    :param beta:
    :param max_iterations: Maximum number of iterations.
    :param tolerance: Tolerance for error.
    :param num_jobs: Number of jobs
    :return: PageRank dict if algorithm converges, -1 otherwise.
    """

    node_list = list(graph.nodes())

    # Get transition matrix
    M = get_transition_matrix(graph, node_list).tocsr()

    number_of_nodes = graph.number_of_nodes()

    # PageRank numpy array initialization
    v = np.full((number_of_nodes, 1), 1. / number_of_nodes)

    with Parallel(n_jobs=num_jobs, require='sharedmem') as parallel:
        for i in range(0, max_iterations):
            # Get a copy of the PageRank array
            v_old = v.copy()

            results = parallel(delayed(_page_rank_update)(graph, transition_matrix=M, beta=beta, nodes=X, v=v)
                               for X in chunks(graph.nodes(), math.ceil(len(graph.nodes()) / num_jobs)))

            # Get L1 norm for the PageRank values
            error = np.absolute(v - v_old).sum()

            # Check if the error is less than the tolerance
            if error < number_of_nodes * tolerance:
                # Insert solution in a dictionary with nodes as keys and PageRank as value
                page_rank_dict = {}
                for node, el in zip(node_list, range(0, number_of_nodes)):
                    page_rank_dict[node] = v[el][0]
                return page_rank_dict
        print("PageRank algorithm does not converge!")
        return -1
