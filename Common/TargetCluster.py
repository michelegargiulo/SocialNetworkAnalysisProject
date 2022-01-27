import os
import pandas as pd

dir_name = os.path.dirname(__file__) + '/..'
target_dataset_path = os.path.join(dir_name, 'Dataset/musae_facebook_target.csv')


def read_target_dataset(target_path=target_dataset_path):
    """
    The Function returns a dictionary in which nodes are the keys and the values are their
    real clusters
    :return: Real clusters dictionary
    """
    data = pd.read_csv(target_path)
    string_id = [str(el) for el in list(data.id)]
    return dict(zip(string_id, list(data.page_type)))


def get_real_cluster(node, target_dataset=None):
    """
    This function gets a node and return its real cluster
    :param node: Node id
    :param target_dataset: Dictionary of the target clusters; if the target dataset is not specified,
                           it is read from the corresponding csv
    :return: Real cluster of the node in input
    """
    if target_dataset is None:
        target_dataset = read_target_dataset()
    return target_dataset[node]


def build_target_sets():
    target = read_target_dataset()
    government = set()
    company = set()
    politician = set()
    tvshow = set()
    for key in target:
        if target[key] == "government":
            government.add(key)
        elif target[key] == "company":
            company.add(key)
        elif target[key] == "politician":
            politician.add(key)
        elif target[key] == "tvshow":
            tvshow.add(key)

    return [government, company, politician, tvshow]


def build_target_sets(subgraph):
    target = read_target_dataset()
    government = set()
    company = set()
    politician = set()
    tvshow = set()
    for key in target:
        if key in subgraph.nodes():
            if target[key] == "government":
                government.add(key)
            elif target[key] == "company":
                company.add(key)
            elif target[key] == "politician":
                politician.add(key)
            elif target[key] == "tvshow":
                tvshow.add(key)

    return [government, company, politician, tvshow]
