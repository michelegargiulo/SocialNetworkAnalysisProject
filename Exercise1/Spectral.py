import networkx as nx
from scipy.sparse import linalg


def spectral_four_clusters(G):
    n = G.number_of_nodes()
    nodes = sorted(G.nodes(), key=float)
    # print("Calculating laplacian matrix")
    L = nx.laplacian_matrix(G, nodes).asfptype()  # Laplacian of a graph is a matrix, with diagonal entries being the
    # degree of the corresponding node and off-diagonal entries being -1 if an edge between the corresponding nodes
    # exists and 0 otherwise
    # print(L)  # To see the laplacian of graph uncomment this line The following command computes eigenvalues and
    # eigenvectors of the Laplacian matrix. Recall that these are scalar numbers w_1, ..., w_k and vectors v_1, ...,
    # v_k such that Lv_i=w_iv_i. The first output is the array of eigenvalues in increasing order. The second output
    # contains the matrix of eigenvectors: specifically, the eigenvector of the k-th eigenvalue is given by the k-th
    # column of v
    # print("Calculating eigenvalues")
    w, v = linalg.eigsh(L, n - 1)
    # print(w)  # Print the list of eigenvalues
    # print(v)  # Print the matrix of eigenvectors
    # print(v[:,0])  # Print the eigenvector corresponding to the first returned eigenvalue

    # print("Calculating clusters")
    # Partition in clusters based on the corresponding eigenvector value being positive or negative This is known to
    # return (an approximation of) the sparset cut of the graph That is, the cut with each of the clusters having
    # many edges, and with few edge among clusters Note that this is not the minimum cut (that only requires few edge
    # among clusters, but it does not require many edge within clusters)
    c1 = set()
    c2 = set()
    c3 = set()
    c4 = set()
    for i in range(n):  # This is an O(n) operation, shouldn't take long even on large networks
        if v[i, 0] < 0:
            if v[i, 1] < 0:
                c1.add(nodes[i])
            else:
                c2.add((nodes[i]))
        else:
            if v[i, 1] < 0:
                c3.add(nodes[i])
            else:
                c4.add((nodes[i]))

    return [c1, c2, c3, c4]