from ProfCode.priorityq import PriorityQueue


def get_top_nodes(all_map, num_top):
    pq = PriorityQueue()
    topnodes = dict()
    for node in all_map:
        pq.add(node, -all_map[node])
    while num_top > 0:
        node = pq.pop()
        topnodes[node] = all_map[node]
        num_top -= 1
    return topnodes