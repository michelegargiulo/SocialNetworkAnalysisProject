import networkx as nx
from networkx.readwrite.edgelist import read_edgelist
import os
from NetworkModels import *
from networkx.algorithms.cluster import clustering
from networkx.algorithms.shortest_paths.generic import average_shortest_path_length

dirname = os.path.dirname(__file__)
net_path = os.path.join(dirname, 'net_12')

# Read graph from file
graph = read_edgelist(net_path)
print("The provided graph has %d nodes and %d edges" % (graph.number_of_nodes(), graph.number_of_edges()))

# Clustering coefficient 0.608065
c = list(clustering(graph).values())
avg_c = sum(c) / len(c)
print("The provided graph has clustering coefficient: %f" % avg_c)

# Average shortest path 4.509263
# avg_sp = average_shortest_path_length(graph)
avg_sp = 4.509263
print("The provided graph has average shortest path length: %f" % avg_sp)

# Plot degree distribution of the graph
# scatterplot_degree_distribution(graph)
# hist_degree_distribution(graph)

# Plot degree distribution of a random graph to compare it with the one of the provided graph
# random_graph = randomG(10000, 0.01445)
# print("The random graph has %d nodes and %d edges" % (random_graph.number_of_nodes(), random_graph.number_of_edges()))
# scatterplot_degree_distribution(random_graph)
# hist_degree_distribution(random_graph)


r = 7
k = 35
q = 3.8
new_graph = GenWS2DG(10000, r, k, q)
print("The generated graph with r=%.2f, k=%.2f, q=%.2f has %d nodes and %d edges"
      % (r, k, q, new_graph.number_of_nodes(), new_graph.number_of_edges()))

# Clustering coefficient generated graph
c = list(clustering(new_graph).values())
avg_c_new = sum(c) / len(c)
print("The generated graph has clustering coefficient: %f" % avg_c_new)

# Average shortest path generated graph
avg_sp_new = average_shortest_path_length(new_graph)
print("The generated graph has average shortest path length: %f" % avg_sp_new)

scatterplot_degree_distribution(new_graph)
hist_degree_distribution(new_graph)
