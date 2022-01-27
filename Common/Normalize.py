def normalize(v):
    """
    Function to normalize a numpy array
    :param v: numpy array to normalize
    :return: Normalized array
    """
    scaling = v.max()
    v = v / scaling
    return v