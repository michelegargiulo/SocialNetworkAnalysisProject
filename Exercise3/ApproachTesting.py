from Exercise3.DatasetGeneration import *
from Exercise3.GraphGeneration import *


def check_truthfulness(p1star, p2star, p3star, verbose=False):
    truthful = True
    misreports = 0
    for f in range(5):
        for s in range(5):
            for v in range(5):
                score = (f, s, v)
                hide_service = (f, None, v)
                hide_value = (f, s, None)
                hide_both = (f, None, None)
                if score in p1star:
                    if hide_service in p2star or hide_service in p3star:
                        if verbose:
                            print(score, "in 1 star")
                            print(hide_service)
                        truthful = False
                        misreports += 1
                    if hide_value in p2star or hide_value in p3star:
                        if verbose:
                            print(score, "in 1 star")
                            print(hide_value)
                        truthful = False
                        misreports += 1
                    if hide_both in p2star or hide_both in p3star:
                        if verbose:
                            print(score, "in 1 star")
                            print(hide_both)
                        truthful = False
                        misreports += 1
                elif score in p2star:
                    if verbose:
                        print(score, "in 1 star")
                        print(score, "in 2 star")
                    if hide_service in p3star or hide_value in p3star or hide_both in p3star:
                        if verbose:
                            print(hide_value, hide_service, hide_both)
                        truthful = False
                        misreports += 1
    print("Total misreports: ", misreports)
    return truthful


def setup_tests():

    ## TEST 1: sigma=0.0001
    dataset = generate_dataset(100000, sigma=0.0001)
    p1star, p2star, p3star = run_test(dataset)
    print("Dataset length: 100000; Sigma: e10-4 (almost exact star assignment); Truthfulness:", check_truthfulness(p1star, p2star, p3star))

    ## TEST 2: sigma=0.3
    dataset = generate_dataset(100000, sigma=0.3)
    p1star, p2star, p3star = run_test(dataset)
    print("Dataset length: 100000; Sigma: 0.3; Truthfulness:", check_truthfulness(p1star, p2star, p3star))

    ## TEST 3: sigma=10
    dataset = generate_dataset(100000, sigma=10)
    p1star, p2star, p3star = run_test(dataset)
    print("Dataset length: 100000; Sigma: 10 (almost random star assignment); Truthfulness:", check_truthfulness(p1star, p2star, p3star))

    ## TEST 4: Alternative approach
    dataset = generate_alternative_dataset(100000, 0.05)
    p1star, p2star, p3star = run_test(dataset)
    print("Dataset length: 100000; Probability of random increase/decrease: 5%; Truthfulness:",
    check_truthfulness(p1star, p2star, p3star))


def run_test(data):
    distribution = get_probabilities(data)

    Dplus = {}
    Dminus = {}

    for key in distribution:
        Dplus[key] = 1 - distribution[key][0]  # Probability of having more than 1 star
        Dminus[key] = distribution[key][0]  # Probability of having 1 star

    graph = create_graph(Dminus, Dplus)
    (_, partition) = nx.minimum_cut(graph, 's', 't')
    p_1_star = partition[0]
    p_23_star = partition[1]

    for key in distribution:
        if key not in p_1_star:
            Dplus[key] = distribution[key][2]  # Probability of having 3 stars
            Dminus[key] = distribution[key][1]  # Probability of having 2 stars
        else:
            del Dplus[key]
            del Dminus[key]

    graph = create_graph(Dminus, Dplus)
    (_, partition) = nx.minimum_cut(graph, 's', 't')

    p_2_star = partition[0]
    p_3_star = partition[1]

    return p_1_star, p_2_star, p_3_star
