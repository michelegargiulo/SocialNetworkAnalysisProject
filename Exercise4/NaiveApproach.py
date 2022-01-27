from Exercise3.DatasetGeneration import generate_dataset, get_probabilities
import numpy as np


def naive_approach(data_set = None):

    if data_set is None:
        data_set = generate_dataset(10000)

    probs = get_probabilities(data_set)
    # probs: (food_score, service_score, value_score) -> [P(1_star), P(2_star), P(3_star)]

    results = {}
    for food in [0, 1, 2, 3, 4, 5]:
        for service in [0, 1, 2, 3, 4, 5, None]:
            for value in [0, 1, 2, 3, 4, 5, None]:
                key = (food, service, value)
                results[key] = np.argmax(probs[key]) + 1

    return results
