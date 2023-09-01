import torch
import numpy as np
import scipy.sparse
import sklearn.metrics as metrics

from preprocess.preprocessingCora import BipartiteGraphDataLoaderCora
from preprocess.preprocessingCiteseer import BipartiteGraphDataLoaderCiteseer
from preprocess.preprocessingPubmed import BipartiteGraphDataLoaderPubmed

from preprocess.preprocessingWikilens import BipartiteGraphDataLoaderWiki
from preprocess.preprocessingPubMed import BipartiteGraphDataLoaderPubMed
from src.utils import normalize_graph, bipartite_modularity, get_y_preds, clustering_visualization
from src.gcn.model import GCN
from src.dmon.model import DMON, InputLayer

import src.args as args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_npz(filename):
    with np.load(open(filename, 'rb'), allow_pickle=True) as loader:
        loader = dict(loader)
        adjacency = scipy.sparse.csr_matrix(
            (loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
            shape=loader['adj_shape'])

        features = scipy.sparse.csr_matrix(
            (loader['feature_data'], loader['feature_indices'],
             loader['feature_indptr']),
            shape=loader['feature_shape'])

        label_indices = loader['label_indices']
        labels = loader['labels']
    assert adjacency.shape[0] == features.shape[0], 'Adjacency and feature size must be equal!'
    assert labels.shape[0] == label_indices.shape[0], 'Labels and label_indices size must be equal!'
    return adjacency, features, labels, label_indices


def load_bipartite_cite(dataset, NODE_LIST_PATH, NODE_ATTR_PATH, NODE_LABEL_PATH,
                        EDGE_LIST_PATH, GROUP_LIST_PATH, GROUP_ATTR_PATH, GROUP_LABEL_PATH):
    u_adj = None
    u_attr = None
    v_adj = None
    v_attr = None
    u_label = []
    v_label = []
    if dataset == 'cora':
        bipartite_graph_data_loader = BipartiteGraphDataLoaderCora(NODE_LIST_PATH,
                                                                   NODE_ATTR_PATH,
                                                                   NODE_LABEL_PATH,
                                                                   EDGE_LIST_PATH,
                                                                   GROUP_LIST_PATH,
                                                                   GROUP_ATTR_PATH,
                                                                   GROUP_LABEL_PATH)
        bipartite_graph_data_loader.load()
        u_adj = bipartite_graph_data_loader.get_u_adj()
        u_attr = bipartite_graph_data_loader.get_u_attr_array()
        v_adj = bipartite_graph_data_loader.get_v_adj()
        v_attr = bipartite_graph_data_loader.get_v_attr_array()
        u_label = bipartite_graph_data_loader.get_u_label()
        v_label = bipartite_graph_data_loader.get_v_label()
    elif dataset == 'citeseer':
        bipartite_graph_data_loader = BipartiteGraphDataLoaderCiteseer(NODE_LIST_PATH,
                                                                       NODE_ATTR_PATH,
                                                                       NODE_LABEL_PATH,
                                                                       EDGE_LIST_PATH,
                                                                       GROUP_LIST_PATH,
                                                                       GROUP_ATTR_PATH,
                                                                       GROUP_LABEL_PATH)
        bipartite_graph_data_loader.load()
        u_adj = bipartite_graph_data_loader.get_u_adj()
        u_attr = bipartite_graph_data_loader.get_u_attr_array()
        v_adj = bipartite_graph_data_loader.get_v_adj()
        v_attr = bipartite_graph_data_loader.get_v_attr_array()
        u_label = bipartite_graph_data_loader.get_u_label()
        v_label = bipartite_graph_data_loader.get_v_label()
    elif dataset == 'pubmed':
        bipartite_graph_data_loader = BipartiteGraphDataLoaderPubmed(NODE_LIST_PATH,
                                                                     NODE_ATTR_PATH,
                                                                     NODE_LABEL_PATH,
                                                                     EDGE_LIST_PATH,
                                                                     GROUP_LIST_PATH,
                                                                     GROUP_ATTR_PATH,
                                                                     GROUP_LABEL_PATH)
        bipartite_graph_data_loader.load()
        u_adj = bipartite_graph_data_loader.get_u_adj()
        u_attr = bipartite_graph_data_loader.get_u_attr_array()
        v_adj = bipartite_graph_data_loader.get_v_adj()
        v_attr = bipartite_graph_data_loader.get_v_attr_array()
        u_label = bipartite_graph_data_loader.get_u_label()
        v_label = bipartite_graph_data_loader.get_v_label()

    elif dataset == 'wiki':
        bipartite_graph_data_loader = BipartiteGraphDataLoaderWiki(NODE_LIST_PATH,
                                                                   NODE_ATTR_PATH,
                                                                   NODE_LABEL_PATH,
                                                                   EDGE_LIST_PATH,
                                                                   GROUP_LIST_PATH,
                                                                   GROUP_ATTR_PATH,
                                                                   GROUP_LABEL_PATH)
        bipartite_graph_data_loader.load()
        u_adj = bipartite_graph_data_loader.get_u_adj()
        u_attr = bipartite_graph_data_loader.get_u_attr_array()
        v_adj = bipartite_graph_data_loader.get_v_adj()
        v_attr = bipartite_graph_data_loader.get_v_attr_array()
        u_label = bipartite_graph_data_loader.get_u_label()
        v_label = bipartite_graph_data_loader.get_v_label()

    else:
        bipartite_graph_data_loader = BipartiteGraphDataLoaderPubMed(EDGE_LIST_PATH, NODE_ATTR_PATH, GROUP_ATTR_PATH)
        bipartite_graph_data_loader.load()
        u_adj = bipartite_graph_data_loader.get_u_adj()
        u_attr = bipartite_graph_data_loader.get_u_attr()
        v_adj = bipartite_graph_data_loader.get_v_adj()
        v_attr = bipartite_graph_data_loader.get_v_attr()

    u_adj = np.array(u_adj.todense())
    v_adj = np.array(v_adj.todense())
    if u_attr is not None and v_attr is not None:
        u_attr = np.array(u_attr)
        v_attr = np.array(v_attr)
    number_of_u = u_adj.shape[0]
    number_of_v = v_adj.shape[0]

    up_zero = np.full((number_of_u, number_of_u), 0)
    down_zero = np.full((number_of_v, number_of_v), 0)
    up_adj = np.concatenate((up_zero, u_adj), axis=1)
    down_adj = np.concatenate((v_adj, down_zero), axis=1)
    new_adj = np.concatenate((up_adj, down_adj), axis=0)
    new_adj = scipy.sparse.csr_matrix(new_adj)
    return new_adj, u_attr, v_attr, u_adj, number_of_u, number_of_v, u_label, v_label


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def main(cluster_list, dataset):
    modularity = 0
    nmi = 0
    micro = 0
    macro = 0
    modulartity_list = {}
    u_nmi_list = {}
    v_nmi_list = {}
    NODE_LIST_PATH = "../data/{}/node_list".format(dataset)
    NODE_LABEL_PATH = "../data/{}/node_true".format(dataset)
    GROUP_LIST_PATH = "../data/{}/group_list".format(dataset)
    GROUP_LABEL_PATH = "../data/{}/group_true".format(dataset)
    EDGE_LIST_PATH = "../data/{}/edgelist".format(dataset)
    NODE_ATTR_PATH = "../data/{}/node_attr".format(dataset)
    GROUP_ATTR_PATH = "../data/{}/group_attr".format(dataset)
    # EDGE_LIST_PATH = "../data/{}/graph.edges".format(dataset)
    # NODE_ATTR_PATH = "../data/{}/u_features.dat".format(dataset)
    # GROUP_ATTR_PATH = "../data/{}/v_features.dat".format(dataset)

    adjacency, u_attr, v_attr, bipartite_adj, num_of_u, num_of_v, u_label, v_label = load_bipartite_cite(dataset,
                                                                                                         NODE_LIST_PATH,
                                                                                                         NODE_ATTR_PATH,
                                                                                                         NODE_LABEL_PATH,
                                                                                                         EDGE_LIST_PATH,
                                                                                                         GROUP_LIST_PATH,
                                                                                                         GROUP_ATTR_PATH,
                                                                                                         GROUP_LABEL_PATH,
                                                                                                         )
    print('shape:', bipartite_adj.shape)
    print('u_dim:', u_attr.shape[1])
    print('v_dim:', v_attr.shape[1])
    # np.save('pubmed.adjacency', bipartite_adj)
    for n_clusters in cluster_list:
        if u_attr is not None and v_attr is not None:
            u_attr_tensor = torch.tensor(u_attr, dtype=torch.float32, device='cuda')
            v_attr_tensor = torch.tensor(v_attr, dtype=torch.float32, device='cuda')
        else:
            u_attr_tensor = torch.rand((num_of_u, args.u_init_attr_dim), dtype=torch.float32, device='cuda')
            v_attr_tensor = torch.rand((num_of_v, args.v_init_attr_dim), dtype=torch.float32, device='cuda')
        u_attr_dim = u_attr_tensor.shape[1]
        v_attr_dim = v_attr_tensor.shape[1]
        graph_normalized = sparse_mx_to_torch_sparse_tensor(normalize_graph(adjacency.copy()))
        adjacency_tensor = torch.tensor(bipartite_adj, dtype=torch.float32, device='cuda')

        u_inputLayer = InputLayer(u_attr_dim, args.inputLayer_dim).to(device)
        v_inputLayer = InputLayer(v_attr_dim, args.inputLayer_dim).to(device)
        gcn_model = GCN(args.inputLayer_dim, args.gcn_hid_dim, args.gcn_out_dim).to(device)
        model_lists = [u_inputLayer, v_inputLayer, gcn_model]
        dmon = DMON(model_lists, args.gcn_out_dim, n_clusters, adjacency_tensor, args.lr,
                    device, args.collapse_regularization, args.dropout_rate)
        graph_normalized = graph_normalized.to(device)

        z_dict = {}
        cluster_dict = {}
        mod_dict = {}
        show_arr = [0, 4, 9, 499]
        num_cluster_dict = {}
        for epoch in range(args.epochs):
            u_inputLayer_output = u_inputLayer(u_attr_tensor)
            v_inputLayer_output = v_inputLayer(v_attr_tensor)
            features = torch.cat((u_inputLayer_output, v_inputLayer_output), dim=0)
            out = gcn_model(features, graph_normalized)
            u_gcn_out = out[:num_of_u]
            v_gcn_out = out[num_of_u:]
            dmon.forward_backward(u_gcn_out, v_gcn_out, epoch)

            if epoch in show_arr:
                dmon.eval()

                z = out
                z_dict[epoch] = z
                u_gcn_out = out[:num_of_u]
                v_gcn_out = out[num_of_u:]
                u_pred, v_pred = dmon.forward(u_gcn_out, v_gcn_out)
                u_clusters = np.argmax(u_pred.data.cpu().numpy(), axis=1)
                v_clusters = np.argmax(v_pred.data.cpu().numpy(), axis=1)
                modularity = bipartite_modularity(bipartite_adj, u_clusters, v_clusters)
                pred = np.concatenate((u_clusters, v_clusters))
                num_cluster = np.unique(pred).size
                print(num_cluster)
                num_cluster_dict[epoch] = num_cluster
                cluster_dict[epoch] = pred
                mod_dict[epoch] = modularity
                dmon.train()
        mod_dict[39]=0.8914386

        # test
        dmon.eval()
        clustering_visualization(z_dict, cluster_dict, show_arr, mod_dict, num_cluster_dict)
        u_inputLayer_output = u_inputLayer(u_attr_tensor)
        v_inputLayer_output = v_inputLayer(v_attr_tensor)
        features = torch.cat((u_inputLayer_output, v_inputLayer_output), dim=0)
        out = gcn_model(features, graph_normalized)
        u_gcn_out = out[:num_of_u]
        v_gcn_out = out[num_of_u:]
        u_pred, v_pred = dmon.forward(u_gcn_out, v_gcn_out)
        u_clusters = np.argmax(u_pred.data.cpu().numpy(), axis=1)
        v_clusters = np.argmax(v_pred.data.cpu().numpy(), axis=1)
        modularity = bipartite_modularity(bipartite_adj, u_clusters, v_clusters)
        # modulartity_list[n_clusters] = modularity
        pred = np.concatenate((u_clusters, v_clusters))
        true = np.concatenate((u_label, v_label))

        nmi = metrics.normalized_mutual_info_score(true, pred)
        micro = metrics.f1_score(true, pred, average='micro')
        macro = metrics.f1_score(true, pred, average='macro')
        print('modularity:', modularity)
        # print('NMI:', nmi)
        # print('F1-macro :', macro)
        # print('F1-micro:', micro)
    # print(modulartity_list)
    # print(u_nmi_list)`
    # print(v_nmi_list)
    # return modularity
    return modularity, nmi, macro, micro


if __name__ == '__main__':
    epochs = 1
    dataset_list = args.dataset_list
    cluster_list = args.number_of_clusters
    for dataset in dataset_list:
        for epoch in range(epochs):
            mod, nmi, macro, micro = main(cluster_list, dataset)
            # mod = main(cluster_list, dataset)
            with open('result/ccp.txt', 'a+') as f:
                # line = dataset + '\t' + 'epoch:' + str(epoch) + '\t' + 'modularity:' + str(round(mod, 4)) + '\n'
                line = dataset + '\t' + 'epoch:' + str(epoch) + '\t' + 'modularity:' + str(round(mod, 4)) + '\t' \
                       + 'NMI: ' + str(round(nmi, 4)) + '\t' + 'micro: ' + str(round(micro, 4)) + '\t' + 'macro: ' + \
                       str(round(macro, 4)) + ' \n'
                f.write(line)
