import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from collections import defaultdict
import similarity_graphs as sim
import datasets


def A(V1, V2, W) -> float:
    """Compute the sum of the weights of the edges between two sets of vertices. W is the similarity matrix."""
    sum_weight = 0
    for v1 in V1:
        for v2 in V2:
            sum_weight += W[v1, v2]
    return sum_weight


def Q(partition, W) -> float:
    """
    partition: The partition of the vertices into clusters
    W: The similarity matrix

    Returns the modularity of the partition
    """

    # Create a dictionary of clusters so that we can easily access the vertices in each cluster
    clusters = defaultdict(list)
    for vertex, cluster_id in enumerate(partition.labels_):  # map cluster_id to list of vertices in it
        clusters[cluster_id].append(vertex)
    cluster_sets = [set(vertices) for vertices in clusters.values()]  # convert to set of clusters

    volume = np.sum(W)
    V = np.arange(W.shape[0])  # list of all vertices

    sumQ = 0
    for cluster in cluster_sets:
        sumQ += (A(cluster, cluster, W) / volume) - ((A(cluster, V, W) / volume) ** 2)
    return sumQ


def Spectral_Clustering(W, dataset, k_min=2, k_max=8, show_plot=True):
    """
    Input:
        W: The similarity matrix
        dataset: The dataset
        k_min: The minimum number of clusters
        k_max: The maximum number of clusters
        show_plot: Whether to show the plot or not

    Output:
        partition: The best partition P_k (based on modularity Q(P_k))
        max_modularity: The modularity value for the best partition
    """
    # Step 1: Compute the degree matrix and the unnormalized Laplacian
    D = np.diag(np.sum(W, axis=1))  # Create a vector of the sum of the columns of W and convert it to a diagonal matrix
    Lq = D - W  # Unnormalized Laplacian

    # Step 2: Compute the first k_max eigenvectors of the normalized Laplacian
    eigenvalues, eigenvectors = np.linalg.eig(Lq)
    ind = np.argsort(np.linalg.norm(np.reshape(eigenvalues, (1, len(eigenvalues))), axis=0))
    U_K = np.real(eigenvectors[:, ind[:k_max + 1]])  # First k_max eigenvectors

    best_partition = None
    max_modularity = -np.inf

    for k in range(k_min, k_max + 1):
        # 3.1: Extract the first k eigenvectors from U_K
        U_k = U_K[:, :k]  # First k eigenvectors

        # 3.2: Normalize each row of U_k using the l^2-norm (project onto unit hypersphere)
        row_norms = np.linalg.norm(U_k, axis=1, keepdims=True)
        U_k_normalized = U_k / (row_norms + 1e-10)  # Add a small epsilon to avoid division by zero

        # 3.3: Apply K-means to the rows of U_k to obtain partition P_k
        partition = KMeans(n_clusters=k, random_state=42).fit(U_k_normalized)  # Perform k-means on the rows of U_k

        # 3.4: Compute modularity Q(P_k)
        modularity = Q(partition, W)  # Compute the modularity of the current partition

        # 4: Track the partition with the maximum modularity
        if modularity > max_modularity:
            max_modularity = modularity
            best_partition = partition

        if show_plot:
            plt.figure(figsize=(8, 6))

            # Plot the points with cluster labels
            plt.scatter(dataset[:, 0], dataset[:, 1], c=partition.labels_, cmap='viridis', s=20, edgecolor='k')
            plt.title(f'Clustering result for k = {k}, Modularity = {modularity:.4f}')
            plt.colorbar(label='Cluster')
            plt.show()

        print(f'k = {k}, Modularity = {modularity:.4f}')

    # Step 6: Return the best partition and its corresponding modularity
    return best_partition, max_modularity


def Spectral_Clustering_without_sim(dataset, k_min=2, k_max=8, sim_builder=sim.FullyConnectSim(0.02), show_plot=True):
    """
    Input:
        dataset: [n_samples, n_samples] numpy array if is_adj=True, or, a [n_samples_a, n_features] array otherwise;
        is_adj: boolean, Indicating whether the adjacency matrix is pre-computed. Default: True;
        sim_builder: The convertor of the adjacency matrix to similarity matrix class. Default: FullyConnectSim(0.02).
    Output:
        partition: The best partition P_k (based on modularity Q(P_k))
        max_modularity: The modularity value for the best partition
    """
    # Step 0: Compute the adjacency matrix and the similarity matrix
    adj_mat = squareform(pdist(dataset, metric='euclidean'))
    W = sim_builder.toSim(adj_mat)

    return Spectral_Clustering(W, dataset, k_min, k_max, show_plot)


def plot_results(dataset, partition, modularity):
    plt.figure(figsize=(8, 6))
    plt.scatter(dataset[:, 0], dataset[:, 1], c=partition.labels_, cmap='viridis', s=10)
    plt.title(f'Chosen: k = {partition.n_clusters}, Modularity = {modularity:.4f}')
    plt.colorbar(label='Cluster')
    plt.show()
    print(f'chosen: Modularity = {modularity:.4f}')


def moons_demo(show_plot=True):
    moon_data, W = datasets.moons()
    partition, modularity = Spectral_Clustering(W, moon_data, k_min=2, k_max=7, show_plot=show_plot)
    plot_results(moon_data, partition, modularity)


def circle_demo(show_plot=True):
    circle_data, W = datasets.circles()
    partition, modularity = Spectral_Clustering(W, circle_data, k_min=2, k_max=7, show_plot=show_plot)
    plot_results(circle_data, partition, modularity)


def demo(dataset, sim_builder, show_plot=True):
    spec_re, Qres = Spectral_Clustering_without_sim(dataset,
                                                    sim_builder=sim_builder,
                                                    show_plot=show_plot,
                                                    k_max=6)
    plot_results(dataset, spec_re, Qres)


def toy_graph(show_plot=True):
    data, sim_mat = datasets.toy_graph()
    partition, modularity = Spectral_Clustering(sim_mat, data, k_min=2, k_max=7, show_plot=show_plot)
    plot_results(data, partition, modularity)


def main():
    toy_graph(show_plot=False)
    moons_demo(show_plot=True)
    circle_demo(show_plot=True)
    demo(datasets.blobs(), sim.EpsNeighborSim(2), show_plot=True)
    demo(datasets.gaussian(), sim.KnnSim(20), show_plot=True)


if __name__ == '__main__':
    main()
