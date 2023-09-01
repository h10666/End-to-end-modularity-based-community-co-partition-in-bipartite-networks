import networkx as nx
import numpy as np
from networkx.algorithms.bipartite import biadjacency_matrix


def load_attr(filename):
    attr_dict = {}
    key_list = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            tmp = []
            line.strip()
            u_id = int(line.split('\t')[0])
            attr = line.split('\t')[1]
            for item in attr.split(','):
                tmp.append(float(item))
            attr_dict[u_id] = tmp
            key_list.append(u_id)
    f.close()
    sorted(attr_dict)
    return attr_dict, key_list


class BipartiteGraphDataLoaderPubMed:
    def __init__(self, graph_path, u_path, v_path):
        super(BipartiteGraphDataLoaderPubMed, self).__init__()
        self.graph_path = graph_path
        self.u_path = u_path
        self.v_path = v_path
        self.edges = []
        self.u_nodes = []
        self.v_nodes = []
        self.num_u_nodes = 0
        self.num_v_nodes = 0
        self.num_edges = 0
        self.u_attr_dim = 0
        self.v_attr_dim = 0
        self.u_adjacency_matrix = None
        self.v_adjacency_matrix = None
        self.u_attr = []
        self.v_attr = []

    def load(self):
        u_list = []
        v_list = []
        edges_list = []
        u_attr = []
        v_attr = []

        with open(self.graph_path, 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                u = int(line.strip().split('\t')[0])
                v = int(line.strip().split('\t')[1])
                u_list.append(u)
                v_list.append(v)
                edges_list.append((u, v))
            u_list = sorted(list(set(u_list)))
            v_list = sorted(list(set(v_list)))

            B_u = nx.Graph()
            B_u.add_nodes_from(u_list, bipartite=0)
            B_u.add_nodes_from(v_list, bipartite=1)
            B_u.add_edges_from(edges_list)
            u_adjacency_matrix_np = biadjacency_matrix(B_u, u_list, v_list)
            nodes = B_u.nodes
            B_u.clear()

            B_v = nx.Graph()
            B_v.add_nodes_from(v_list, bipartite=0)
            B_v.add_nodes_from(u_list, bipartite=1)
            B_v.add_nodes_from(edges_list)
            v_adjacency_matrix_np = biadjacency_matrix(B_v, v_list, u_list)
            B_v.clear()

            self.u_nodes = u_list
            self.v_nodes = v_list
            self.edges = edges_list
            self.num_u_nodes = len(u_list)
            self.num_v_nodes = len(v_list)
            self.num_edges = len(edges_list)
            self.u_adjacency_matrix = u_adjacency_matrix_np
            self.v_adjacency_matrix = v_adjacency_matrix_np
        f.close()

        u_attr_dict, u_key_list = load_attr(self.u_path)
        v_attr_dict, v_key_list = load_attr(self.v_path)
        u_del_key = set(u_key_list).difference(set(u_list))
        v_del_key = set(v_key_list).difference(set(v_list))
        for i in u_del_key:
            u_attr_dict.pop(i)
        for j in v_del_key:
            v_attr_dict.pop(j)
        for attr in u_attr_dict.values():
            u_attr.append(attr)
        for attr in v_attr_dict.values():
            v_attr.append(attr)

        u_attr_np = np.array(u_attr, dtype=np.float32)
        v_attr_np = np.array(v_attr, dtype=np.float32)
        self.u_attr = u_attr_np
        self.v_attr = v_attr_np
        self.u_attr_dim = u_attr_np.shape[1]
        self.v_attr_dim = v_attr_np.shape[1]

    def get_u_adj(self):
        return self.u_adjacency_matrix

    def get_v_adj(self):
        return self.v_adjacency_matrix

    def get_num_u_nodes(self):
        return self.num_u_nodes

    def get_num_v_nodes(self):
        return self.num_v_nodes

    def get_u_attr(self):
        return self.u_attr

    def get_v_attr(self):
        return self.v_attr

    def get_u_attr_dim(self):
        return self.u_attr_dim

    def get_v_attr_dim(self):
        return self.v_attr_dim


if __name__ == '__main__':
    graph_path = '../../data/GD/gene_disease_graph.txt'
    u_path = '../../data/GD/gene_NodeAtt.txt'
    v_path = '../../data/GD/disease_NodeAtt.txt'
    loader = BipartiteGraphDataLoaderWiki(graph_path, u_path, v_path)
    loader.load()
    u_adj = loader.get_u_adj()
    print(u_adj)

