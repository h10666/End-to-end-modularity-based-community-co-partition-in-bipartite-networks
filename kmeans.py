import numpy as np
from sklearn.cluster import KMeans
from src.utils import get_y_preds, bipartite_modularity


dataset = 'citeseer'
n_clusters = 16
u_attr_path = 'data/{}/group_attr'.format(dataset)
v_attr_path = 'data/{}/node_attr'.format(dataset)
u_true_path = 'data/{}/group_true'.format(dataset)
v_true_path = 'data/{}/node_true'.format(dataset)

node = []
attr = []
labels = []


def load_attr(file, node_list, attr_list):
    with open(file, 'r') as f:
        for line in f.readlines():
            tmp = []
            line.strip()
            node_id = int(line.split('\t')[0])
            embed = line.split('\t')[1:]
            for item in embed:
                tmp.append(int(item))
            node_list.append(node_id)
            attr_list.append(tmp)
    return node_list, attr_list


def load_label(file, node_label):
    with open(file, 'r') as f:
        for line in f.readlines():
            line.strip()
            label = int(line.split('\t')[1])
            node_label.append(label)
    return node_label


node, attr = load_attr(u_attr_path, node, attr)
node, attr = load_attr(v_attr_path, node, attr)
labels = load_label(u_true_path, labels)
labels = load_label(v_true_path, labels)

attr_np = np.array(attr, dtype=np.float)
clusters = KMeans(n_clusters=n_clusters).fit_predict(attr_np)
y_pred = get_y_preds(clusters, labels, n_clusters)
print(y_pred)


