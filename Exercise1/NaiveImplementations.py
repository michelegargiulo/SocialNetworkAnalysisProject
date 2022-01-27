import random
from scipy.sparse import linalg

import Exercise2.Betweenness
from ProfCode.priorityq import PriorityQueue
import networkx as nx
import numpy as np


def hierarchical(G, targetClusters=4):
    # Create a priority queue with each pair of nodes indexed by distance
    pq = PriorityQueue()
    for u in G.nodes():
        for v in G.nodes():
            if u != v:
                if (u, v) in G.edges() or (v, u) in G.edges():
                    pq.add(frozenset([frozenset(np.array([u])), frozenset(np.array([v]))]), 0)
                else:
                    pq.add(frozenset([frozenset(np.array([u])), frozenset(np.array([v]))]), 1)

    # Start with a cluster for each node
    clusters = set()
    for node in G.nodes():
        clusters.add(frozenset(np.array([node])))

    while len(clusters) > targetClusters:

        # Merge closest clusters
        s = list(pq.pop())
        clusters.remove(s[0])
        clusters.remove(s[1])

        # Update the distance of other clusters from the merged cluster
        for w in clusters:
            e1 = pq.remove(frozenset([s[0], w]))
            e2 = pq.remove(frozenset([s[1], w]))
            if e1 == 0 or e2 == 0:
                pq.add(frozenset(np.array([s[0] | s[1], w])), 0)
            else:
                pq.add(frozenset(np.array([s[0] | s[1], w])), 1)

        clusters.add(s[0] | s[1])

    return clusters


def four_means(G):
    n=G.number_of_nodes()

    # Choose four clusters represented by vertices that are not neighbors
    u = random.choice(list(G.nodes()))
    v = random.choice(list(nx.non_neighbors(G, u)))
    w = random.choice(list(set(nx.non_neighbors(G, u)).intersection(set(nx.non_neighbors(G, v)))))
    z = random.choice(list(set(nx.non_neighbors(G, u)).intersection(set(nx.non_neighbors(G, v))).intersection(set(nx.non_neighbors(G, w)))))

    cluster0 = {u}
    cluster1 = {v}
    cluster2 = {w}
    cluster3 = {z}
    added = 4

    while added < n:
        # Choose a node that is not yet in a cluster and add it to the closest cluster
        l = [el for el in G.nodes() if ((el not in cluster0|cluster1|cluster2|cluster3) and (len(set(G.neighbors(el)).intersection(cluster0)) != 0 or len(set(G.neighbors(el)).intersection(cluster1)) != 0 or len(set(G.neighbors(el)).intersection(cluster2)) != 0 or len(set(G.neighbors(el)).intersection(cluster3)) != 0))]
        x = random.choice(l)

        if len(set(G.neighbors(x)).intersection(cluster0)) != 0:
            cluster0.add(x)
            added+=1

        elif len(set(G.neighbors(x)).intersection(cluster1)) != 0:
            cluster1.add(x)
            added+=1

        elif len(set(G.neighbors(x)).intersection(cluster2)) != 0:
            cluster2.add(x)
            added+=1

        elif len(set(G.neighbors(x)).intersection(cluster3)) != 0:
            cluster3.add(x)
            added+=1

    return[cluster0, cluster1, cluster2, cluster3]


# Computes edge and vertex betweenness of the graph in input
def betweenness(G):
    edge_btw = {frozenset(e): 0 for e in G.edges()}
    node_btw = {i: 0 for i in G.nodes()}

    for s in G.nodes():
        # Compute the number of shortest paths from s to every other node
        tree = []  # it lists the nodes in the order in which they are visited
        spnum = {i: 0 for i in G.nodes()}  # it saves the number of shortest paths from s to i
        parents = {i: [] for i in G.nodes()}  # it saves the parents of i in each of the shortest paths from s to i
        distance = {i:-1 for i in G.nodes()}  # the number of shortest paths starting from s that use the edge e
        eflow = {frozenset(e):0 for e in G.edges()}  # the number of shortest paths starting from s that use the edge e
        vflow = {i: 1 for i in G.nodes()}  # the number of shortest paths starting from s that use the vertex i. It
        # is initialized to 1 because the shortest path from s to i is assumed to uses that vertex once.

        #BFS
        queue = [s]
        spnum[s] = 1
        distance[s] = 0
        while queue != []:
            c = queue.pop(0)
            tree.append(c)
            for i in G[c]:
                if distance[i] == -1:  # if vertex i has not been visited
                    queue.append(i)
                    distance[i] = distance[c]+1
                if distance[i] == distance[c]+1:  # if we have just found another shortest path from s to i
                    spnum[i] += spnum[c]
                    parents[i].append(c)

        # BOTTOM-UP PHASE
        while tree != []:
            c=tree.pop()
            for i in parents[c]:
                eflow[frozenset({c, i})] += vflow[c] * (spnum[i]/spnum[c])  # the number of shortest paths using
                # vertex c is split among the edges towards its parents proportionally to the number of shortest
                # paths that the parents contributes
                vflow[i] += eflow[frozenset({c, i})]  # each shortest path that use an edge (i,c) where i is closest
                # to s than c must use also vertex i
                edge_btw[frozenset({c, i})] += eflow[frozenset({c, i})]  # betweenness of an edge is the sum over all
                # s of the number of shortest paths from s to other nodes using that edge
            if c != s:
                node_btw[c]+=vflow[c]  # betweenness of a vertex is the sum over all s of the number of shortest paths
                # from s to other nodes using that vertex

    return edge_btw, node_btw


# The algorithm is quite time-consuming. Indeed, its computational complexity is O(nm).
# Possible optimizations: parallelization, considering only a sample of starting nodes
# Clusters are computed by iteratively removing edges of largest betweenness
def girvan_newman(graph, numClusters=4):
    # eb,nb = betweenness_centrality_parallel(graph, numJobs)
    eb, nb = Exercise2.Betweenness.betweenness(graph)
    # pq = PriorityQueue()
    # for i in eb.keys():
    #     pq.add(i, -eb[i])
    graph = graph.copy()

    while len(list(nx.connected_components(graph))) < numClusters:
        edge = tuple(max(eb, key=eb.get))
        graph.remove_edges_from([edge])
        eb, nb = Exercise2.Betweenness.betweenness(graph)

    return list(nx.connected_components(graph))


def spectral(G):
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    L = nx.laplacian_matrix(G, nodes).asfptype()  # Laplacian of a graph is a matrix, with diagonal entries being the
    # degree of the corresponding node and off-diagonal entries being -1 if an edge between the corresponding nodes
    # exists and 0 otherwise
    # print(L) #To see the laplacian of graph uncomment this line The following command computes eigenvalues and
    # eigenvectors of the Laplacian matrix. Recall that these are scalar numbers w_1, ..., w_k and vectors v_1, ...,
    # v_k such that Lv_i=w_iv_i. The first output is the array of eigenvalues in increasing order. The second output
    # contains the matrix of eigenvectors: specifically, the eigenvector of the k-th eigenvalue is given by the k-th
    # column of v
    w, v = linalg.eigsh(L, n - 1)
    # print(w) # Print the list of eigenvalues
    # print(v) # Print the matrix of eigenvectors
    # print(v[:,0]) # Print the eigenvector corresponding to the first returned eigenvalue

    # Partition in clusters based on the corresponding eigenvector value being positive or negative This is known to
    # return (an approximation of) the sparset cut of the graph That is, the cut with each of the clusters having
    # many edges, and with few edge among clusters Note that this is not the minimum cut (that only requires few edge
    # among clusters, but it does not require many edge within clusters)
    c1 = set()
    c2 = set()
    for i in range(n):
        if v[i, 0] < 0:
            c1.add(nodes[i])
        else:
            c2.add(nodes[i])
    return (c1, c2)

# How to achieve more than two clusters? Two options: (i) for each subgraph corresponding to one of the clusters,
# we can split this subgraph by running the spectral algorithm on it; (ii) we can use further eigenvectors. For
# example, we can partition nodes in four clusters by using the first two eigenvectors, so that the first (second,
# respectively) cluster contains those nodes i such that v[i,0] and v[i,1] are both negative (both non-negative,
# resp.) while the third (fourth, respectively) cluster contains those nodes i such that only v[i,0] (only v[i,1],
# resp.) is negative.
