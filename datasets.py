import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs, make_moons, make_gaussian_quantiles, make_circles
import similarity_graphs as sim


def toy_graph():
    points = [
        (3, 7),
        (4.5225, 7.42125),
        (2.4425, 6.02125),
        (3.5625, 5.38125),
        (5.0625, 5.86125),
        (3.0025, 0.80125),
        (0.7425, -1.01875),
        (1.7825, -0.89875),
        (3.6425, -1.45875),
        (4.6425, -0.17875),
        (7.3425, 1.78125),
        (8.3425, -0.47875),
        (8.8025, 1.04125),
        (9.4425, 2.42125),
        (10.6425, 0.16125)
    ]
    data = np.array(points)

    sim_mat = np.array([
        [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    ])

    return data, sim_mat


def blobs():
    blob_data, _ = make_blobs(n_samples=1000, centers=4, cluster_std=1.0, random_state=42)
    return blob_data


def gaussian():
    gaussian_data, _ = make_gaussian_quantiles(mean=None, cov=0.5, n_samples=1000, n_features=2, n_classes=3, random_state=42)
    return gaussian_data


def moons():
    moon_data, moon_labels = make_moons(1000, noise=0.05)
    adj_mat = squareform(pdist(moon_data, metric='euclidean'))
    W = sim.FullyConnectSim(0.5).toSim(adj_mat)

    for i in range(1000):
        for j in range(1000):
            if moon_labels[i] != moon_labels[j]:
                W[i, j] = 0
    np.fill_diagonal(W, 0)

    return moon_data, W


def circles():
    circle_data, circle_labels = make_circles(n_samples=1000, factor=0.5, noise=0.05)
    adj_mat = squareform(pdist(circle_data, metric='euclidean'))
    W = sim.FullyConnectSim(0.5).toSim(adj_mat)

    for i in range(1000):
        for j in range(1000):
            if circle_labels[i] != circle_labels[j]:
                W[i, j] = 0
    np.fill_diagonal(W, 0)

    return circle_data, W
