from abc import ABC, abstractmethod

import numpy as np


class SimilarityGraphBuilder(ABC):
    @abstractmethod
    def toSim(self, adj_mat):
        pass


class FullyConnectSim(SimilarityGraphBuilder):
    def __init__(self, sigma):
        self.sigma = sigma

    def toSim(self, adj_mat):
        return np.exp(-adj_mat / (2 * self.sigma))


class EpsNeighborSim(SimilarityGraphBuilder):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def toSim(self, adj_mat):
        return (adj_mat <= self.epsilon).astype('float64')


class KnnSim(SimilarityGraphBuilder):
    def __init__(self, k):
        self.k = k

    def toSim(self, adj_mat):
        W = np.zeros(adj_mat.shape)
        # Sort the adjacency matrx by rows and record the indices
        Adj_sort = np.argsort(adj_mat, axis=1)

        # Set the weight (i,j) to 1 when either i or j is within the k-nearest neighbors of each other
        for i in range(Adj_sort.shape[0]):
            W[i, Adj_sort[i, :][:(self.k + 1)]] = 1

        return W


class MutualKnnSim(SimilarityGraphBuilder):
    def __init__(self, k):
        self.k = k

    def toSim(self, adj_mat):
        W1 = np.zeros(adj_mat.shape)
        # Sort the adjacency matrx by rows and record the indices
        Adj_sort = np.argsort(adj_mat, axis=1)
        # Set the weight W1[i,j] to 0.5 when either i or j is within the k-nearest neighbors of each other (Flag)
        # Set the weight W1[i,j] to 1 when both i and j are within the k-nearest neighbors of each other
        for i in range(adj_mat.shape[0]):
            for j in Adj_sort[i, :][:(self.k + 1)]:
                if i == j:
                    W1[i, i] = 1
                elif W1[i, j] == 0 and W1[j, i] == 0:
                    W1[i, j] = 0.5
                else:
                    W1[i, j] = W1[j, i] = 1
        return np.copy((W1 > 0.5).astype('float64'))
