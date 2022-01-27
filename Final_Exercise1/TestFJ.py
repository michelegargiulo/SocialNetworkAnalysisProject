from Final_Exercise1.FJ_Dynamics import FJ_dynamics
from Common.DynamicGraphGeneration import get_random_graph
import random
import networkx as nx
import numpy as np
import math

l = list(np.linspace(0.1, 0.6, 1000))

for i in l:
    graph = get_random_graph(1000, math.ceil((1000 * (1000 - 1)) * 0.5 * i))
    b = {}
    s = {}

    for v in graph.nodes():
        b[v] = random.uniform(0, 1)
        s[v] = random.uniform(0, 1)

    if FJ_dynamics(graph, b, s) == -1:
        print("Algorithm does not converge!")
    else:
        print("Ok!")
