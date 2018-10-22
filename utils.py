import numpy as np

def print_k_nearest_neighbour(X, idx, k, idx_to_word):
    """
    :param X: Embedding Matrix |V x D|
    :param idx: The Knn of the ith
    :param k: k nearest neighbour
    :return:
    """

    dists = np.dot((X - X[idx]) ** 2, np.ones(X.shape[1]))
    idxs = np.argsort(dists)[:k]

    print('The {} nearest neighbour of {} are: '.format(str(k), idx_to_word[idx]))
    for i in idxs:
        print(idx_to_word[i])
    print("====================")
    return idxs

