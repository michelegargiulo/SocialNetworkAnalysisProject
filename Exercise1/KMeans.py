import random
import networkx as nx

''''#####################################################################################################################
The idea of the optimization, came from the observation that the naive implementation iterates over the nodes, for each
node, taking O(N^2). For a bigger graph, doing a nested iteration over nodes is not feasible.

Since we have to assign a cluster to each node, we could in theory visit each node only once
The idea is to exploit the constant access time of an hashmap, adding and updating associations between node->cluster
as new nodes are discovered and clusters are assigned

The algorithm starts from a list called "nodeList" which contains all the nodes except the four starter nodes
The algorithm tracks the neighbors of the 4 starters in a list called "choices"
The algorithm chooses randomly a node called "curNode" from "choices" and examines its neighbors, taking O(k), 
    where k is the number of neighbors
When a neighbor which has an assigned cluster is found, assign "curNode" to the same cluster
For all the other neighbors:
    If they are discovered (present in hashmap), ignore them
    Otherwise, "discover" them, by adding them to "choices" and to the hashmap as UNASSIGNED
Remove "curNode" from "choices" (since it has now been visited) and remove it from "nodeList"
When all nodes have been visited, "nodeList" is empty (so the algorithm exits the while loop) and the hashmap contains
    a map from all the nodes to their cluster
Last step is to build the cluster as a set/list by iterating over the hashmap (O(N)) and reading the labels
The algorithm returns the cluster

ANALYSIS:
#####################################################################################################################'''

# Just labels for the various clusters
CLUSTER_0 = 0
CLUSTER_1 = 1
CLUSTER_2 = 2
CLUSTER_3 = 3
UNASSIGNED = -1


def four_means_optimized(G):
    nodeList = list(G.nodes())  # Take a copy of the graph nodes, taking O(N)
    choices = []
    hmap = {}

    # Choose four clusters represented by vertices that are non-neighbors
    u = random.choice(list(G.nodes()))
    v = random.choice(list(nx.non_neighbors(G, u)))
    w = random.choice(list(set(nx.non_neighbors(G, u)).intersection(set(nx.non_neighbors(G, v)))))
    z = random.choice(list(set(nx.non_neighbors(G, u)).intersection(set(nx.non_neighbors(G, v))).intersection(
        set(nx.non_neighbors(G, w)))))

    # Initialize the clusters and add the nodes and their corresponding cluster label to "hmap"
    cluster = [{u}, {v}, {w}, {z}]
    hmap[u] = CLUSTER_0
    hmap[v] = CLUSTER_1
    hmap[w] = CLUSTER_2
    hmap[z] = CLUSTER_3

    # Remove those 4 nodes from "nodeList", since they cannot be a valid choice
    nodeList.remove(u)
    nodeList.remove(v)
    nodeList.remove(w)
    nodeList.remove(z)

    # Get the nodes which are adjacent to at least to one of the four members of the cluster
    choices = list(
        set().union(nx.neighbors(G, u)).union(nx.neighbors(G, v)).union(nx.neighbors(G, w)).union(nx.neighbors(G, z)))
    for node in choices:  # Add such nodes to "hmap" and set them as UNASSIGNED
        hmap[node] = UNASSIGNED

    # Assign the clusters using k-means
    while len(nodeList) > 0:  # Iterate on each node only once taking O(N)
        # For each node of nodeList, select one which is a valid choice
        # (neighbor to a node which has an assigned cluster
        curNode = random.choice(choices)

        # The not is not eligible anymore as a choice, since it will soon have an assigned cluster
        choices.remove(curNode)

        # Iterate over its neighbors
        for neighbor in list(nx.neighbors(G, curNode)):

            # If we are still searching for a neighbor with an assigned cluster, AND
            # if the neighbor is in hmap (so it is either assigned or unassigned, but already discovered), AND
            # if the node is unassigned, then:
            if (neighbor in hmap) and (hmap[neighbor] != UNASSIGNED):
                # If the current node is unassigned, then assign to the same cluster of its neighbor
                if hmap[curNode] == UNASSIGNED:
                    hmap[curNode] = hmap[neighbor]
                    # Remove the node from nodeList, since it cannot be chosen again
                    nodeList.remove(curNode)

            # Otherwise, if we already assigned "curNode" to a cluster, or the neighbor is not in hmap (so it is
            # undiscovered, or it was unassigned:
            else:
                # If the neighbor is eligible to be a choice, because it was not discovered, or discovered but
                # unassigned, it means that it is now a valid choice, and if it is not present in choices, then add it
                if neighbor in nodeList and neighbor not in choices:
                    choices.append(neighbor)
                    hmap[neighbor] = UNASSIGNED

    # Get the clusters from "hmap". This iterates over the map in O(N)
    for curNode in hmap:
        if hmap[curNode] == CLUSTER_0:
            cluster[0].add(curNode)
        if hmap[curNode] == CLUSTER_1:
            cluster[1].add(curNode)
        if hmap[curNode] == CLUSTER_2:
            cluster[2].add(curNode)
        if hmap[curNode] == CLUSTER_3:
            cluster[3].add(curNode)

    # Return the clusters found
    return cluster
