import numpy as np
import scipy.sparse
import sklearn
from munkres import Munkres
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


def normalize_graph(graph, normalized=True, add_self_loops=True):
    if add_self_loops:
        graph = graph + scipy.sparse.identity(graph.shape[0])
    degree = np.squeeze(np.asarray(graph.sum(axis=1)))
    if normalized:
        with np.errstate(divide='ignore'):
            inverse_sqrt_degree = 1. / np.sqrt(degree)
        inverse_sqrt_degree[inverse_sqrt_degree == np.inf] = 0
        inverse_sqrt_degree = scipy.sparse.diags(inverse_sqrt_degree)
        return inverse_sqrt_degree @ graph @ inverse_sqrt_degree
    else:
        with np.errstate(divide='ignore'):
            inverse_degree = 1. / degree
        inverse_degree[inverse_degree == np.inf] = 0
        inverse_degree = scipy.sparse.diags(inverse_degree)
        return inverse_degree @ graph


def modularity(adjacency, clusters):
    """Computes graph modularity.

  Args:
    adjacency: Input graph in terms of its sparse adjacency matrix.
    clusters: An (n,) int cluster vector.

  Returns:
    The value of graph modularity.
    https://en.wikipedia.org/wiki/Modularity_(networks)
  """
    degrees = adjacency.sum(axis=0).A1
    n_edges = degrees.sum()  # Note that it's actually 2*n_edges.
    result = 0
    for cluster_id in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        adj_submatrix = adjacency[cluster_indices, :][:, cluster_indices]
        degrees_submatrix = degrees[cluster_indices]
        result += np.sum(adj_submatrix) - (np.sum(degrees_submatrix) ** 2) / n_edges
    return result / n_edges


def bipartite_modularity(adjacency, u_clusters, v_clusters):
    result = 0
    total_num_edges = float(np.sum(adjacency))
    clusters = np.concatenate((u_clusters, v_clusters))
    u_degrees = np.sum(adjacency, axis=1)
    v_degrees = np.sum(adjacency, axis=0)
    for clusters_id in np.unique(clusters):
        u_clusters_indices = np.where(u_clusters == clusters_id)[0]
        v_clusters_indices = np.where(v_clusters == clusters_id)[0]
        cluster_sum_edges = 0
        u_sum_degrees = 0
        v_sum_degrees = 0

        for i in u_clusters_indices:
            for j in v_clusters_indices:
                if adjacency[i][j] != 0:
                    cluster_sum_edges += 1
        u_sum_degrees += np.sum(u_degrees[u_clusters_indices])
        v_sum_degrees += np.sum(v_degrees[v_clusters_indices])

        behind = u_sum_degrees * v_sum_degrees / total_num_edges
        result += (cluster_sum_edges - behind)
    return result / total_num_edges


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    clusterLabels = np.zeros(n_clusters)
    for i in range(n_clusters):
        clusterLabels[i] = indices[i][1]
    return clusterLabels


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))

    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_y_preds(cluster_assignments, y_true, n_clusters):
    '''
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)

    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset

    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    '''
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred

def clustering_visualization(z_dict, label_dict, show_arr, mod_dict, n_cluster_dict):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    sns.set_style('darkgrid')
    # sns.set_palette('muted')
    sns.set_context('paper', font_scale=1.5, rc={'lines.linewidth': 2})

    # palette = sns.color_palette('bright', 6)
    for i, epoch in enumerate(show_arr):
        z = z_dict[epoch].data.cpu().numpy()
        label = label_dict[epoch]
        X_tsne = TSNE(n_components=2, random_state=313).fit_transform(z)
        sns.scatterplot(X_tsne[:, 0], X_tsne[:, 1], hue=label, marker='+',
                        palette=sns.color_palette('bright', n_cluster_dict[epoch]), ax=axes[i], legend=False)
        axes[i].set_title(f'({i + 1}) epoch {show_arr[i]+1} (Modularity = {mod_dict[show_arr[i]]:.3f})',
                          font='Times New Roman', fontsize=12)
    fig.savefig('citeseer.pdf', dpi=600, format='pdf')
    plt.show()