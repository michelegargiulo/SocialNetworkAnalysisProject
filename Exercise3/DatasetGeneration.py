import random
import os
import pandas as pd


dirname = os.path.dirname(__file__)
restourant_path = os.path.join(dirname, "../Dataset/restaurants.csv")


def read_dataset(hidden_value=None, data=None):
    """
    Reads a dataset from a csv file, replacing all the -1 found in the data with hiddenValue (default: None)
    :param hidden_value:
    :return:
    """
    if data is None:
        data = pd.read_csv(restourant_path).values.tolist()

    for elem in data:
        for i, value in enumerate(elem):
            if value == -1:
                elem[i] = hidden_value
    return data


def generate_write_dataset(length=1000, sigma=0.3):
    """
    Generates a dataset which contains 3 features: Food, Service and Value, and 1 label: Stars
    Food can go from 0 to 5 (inclusive)
    Service and Value can go from 0 to 5 (inclusive) or be hidden (-1)
    A number of stars is awarded in correlation to the total score. Hiding a feature decreases the probability of having
    more stars. Minimum score is -2 [0, -1, -1], maximum is 15 [5, 5, 5]
    :param length:
    :return:
    """
    dataset = []
    for _ in range(length):
        # Generate a random score for the 3 features
        food = random.randint(0, 5)
        service = random.randint(-1, 5)
        value = random.randint(-1, 5)

        # Assign a Michelen score based on feature score
        # Minimum score is -2, maximum score is 15
        # -2 -> 1, 15 -> 3
        m = 3 / 5
        q = 1 / 2
        x = food
        num = 1
        if service >= 0:
            x += service
            num += 1
        if value >= 0:
            x += value
            num += 1
        x = x / num
        mean = m * x + q
        # sigma = 0.3 by default

        stars = round(random.gauss(mean, sigma))
        if stars < 1:
            stars = 1
        if stars > 3:
            stars = 3

        food = int(food)
        service = int(service)
        value = int(value)
        dataset.append((food, service, value, stars))

    df = pd.DataFrame(dataset, columns=['Food', 'Service', 'Value', 'Stars'])
    df.to_csv(restourant_path, index=False)


def generate_dataset(length=1000, sigma=0.3):
    """
    Generates a dataset which contains 3 features: Food, Service and Value, and 1 label: Stars
    Food can go from 0 to 5 (inclusive)
    Service and Value can go from 0 to 5 (inclusive) or be hidden (-1)
    A number of stars is awarded in correlation to the total score. Hiding a feature decreases the probability of having
    more stars. Minimum score is -2 [0, -1, -1], maximum is 15 [5, 5, 5]
    :param length:
    :return:
    """
    dataset = []
    for _ in range(length):
        # Generate a random score for the 3 features
        food = random.randint(0, 5)
        service = random.randint(-1, 5)
        value = random.randint(-1, 5)

        # Assign a Michelen score based on feature score
        # Minimum score is 0, maximum score is 15. Hidden features are ignored in the computing of the average score
        # -2 -> 1, 15 -> 3
        m = 3 / 5
        q = 1 / 2
        x = food
        num = 1
        if service >= 0:
            x += service
            num += 1
        if value >= 0:
            x += value
            num += 1
        x = x / num
        mean = m * x + q
        # sigma = 0.3 by default

        stars = round(random.gauss(mean, sigma))
        if stars < 1:
            stars = 1
        if stars > 3:
            stars = 3

        food = int(food)
        service = int(service)
        value = int(value)
        dataset.append((food, service, value, stars))

    return dataset


def generate_alternative_dataset(length=1000, probability=0.05):
    """
    Generates a dataset which contains 3 features: Food, Service and Value, and 1 label: Stars
    Food can go from 0 to 5 (inclusive)
    Service and Value can go from 0 to 5 (inclusive) or be hidden (-1)
    A number of stars is awarded in correlation to the total score. Hiding a feature decreases the probability of having
    more stars. Minimum score is -2 [0, -1, -1], maximum is 15 [5, 5, 5]
    :param length:
    :return:
    """
    dataset = []

    for _ in range(length):
        # Generate a random score for the 3 features
        food = random.randint(0, 5)
        service = random.randint(-1, 5)
        value = random.randint(-1, 5)

        # Assign a Michelen score based on feature score
        # Minimum score is 0, maximum score is 15. Hidden features are ignored in the computing of the average score
        # -2 -> 1, 15 -> 3
        m = 3 / 5
        q = 1 / 2
        x = food
        num = 1
        if service >= 0:
            x += service
            num += 1
        if value >= 0:
            x += value
            num += 1
        x = x / num
        mean = m * x + q

        stars = round(mean)

        choice = random.uniform(0, 1)
        if choice <= probability:
            stars += 1
            if random.uniform(0, 1) <= probability:
                stars += 1
        elif 1 - choice <= probability:
            stars -= 1
            if random.uniform(0, 1) <= probability:
                stars -= 1

        if stars < 1:
            stars = 1
        if stars > 3:
            stars = 3

        food = int(food)
        service = int(service)
        value = int(value)
        dataset.append((food, service, value, stars))

    return dataset


def get_probabilities(data, subWithNone=False):
    """
    Loads the probability distribution of the dataset.
    Returns a map from scores -> probability of having a certain number of stars
    For example, if  distribution[(3, 4, None)] = [0.25, 0.5, 0.25]
    It means that for a Food score of 3 (out of 5), service score of 4 and a value score omitted,
    there is a probability of 25% of having 1 star, 50% of having 2 and 25% of having 3
    :return:
    """
    distribution = {}
    for row in data:
        tup = (row[0], row[1], row[2])
        if tup not in distribution:
            if subWithNone:
                s = row[1]
                v = row[2]
                if row[1] == -1:
                    s = None
                if row[2] == -1:
                    v = None
                tup = (tup[0], s, v)
            distribution[tup] = [0, 0, 0]
            distribution[tup][row[3]-1] += 1
        else:
            distribution[tup][row[3]-1] += 1

    # Normalize so they sum up to 1
    for key in distribution:
        distribution[key] = [float(i)/sum(distribution[key]) for i in distribution[key]]

    return distribution
