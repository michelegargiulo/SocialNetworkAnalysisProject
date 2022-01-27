import networkx as nx


def create_graph(minus_distr, plus_distr):
    """
    Creates the graph on which to apply the mincut
    :param minus_distr:
    :param plus_distr:
    :return:
    """
    graph = nx.DiGraph()
    for comb in minus_distr:
        w = minus_distr[comb]
        graph.add_edge('s', comb, capacity=w)
    for comb in plus_distr:
        w = plus_distr[comb]
        graph.add_edge(comb, 't', capacity=w)
    graph = create_inf_edges(graph)
    return graph


def create_inf_edges(G):
    """
    Sets to infinite the weights of every edge that starts from a node that has a certain number of stars
    and ends in another node, that hides a feature and has a strictly greater number of stars
    :param G:
    :return:
    """
    for node in G.nodes():
        if node != 's' and node != 't':
            if node[1] is not None and node[2] is not None:
                if (node[0], None, None) in G.nodes():
                    G.add_edge(node, (node[0], None, None), capacity=float('inf'))
                if (node[0], node[1], None) in G.nodes():
                    G.add_edge(node, (node[0], node[1], None), capacity=float('inf'))
                if (node[0], None, node[2]) in G.nodes():
                    G.add_edge(node, (node[0], None, node[2]), capacity=float('inf'))
            if node[1] is None and node[2] is not None and (node[0], None, None) in G.nodes():
                G.add_edge(node, (node[0], None, None), capacity=float('inf'))
            if node[1] is not None and node[2] is None and (node[0], None, None) in G.nodes():
                G.add_edge(node, (node[0], None, None), capacity=float('inf'))
    return G
