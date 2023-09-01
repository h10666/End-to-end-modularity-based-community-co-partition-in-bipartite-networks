import torch
import dgl
import numpy as np
import networkx as nx

G = nx.Graph()
G.add_nodes_from([12, 13, 23, 24, 329, 431])
edges_list = [(12, 13), (13, 24), (329, 431)]
G.add_edges_from(edges_list)
net = dgl.from_networkx(G)
# net.add_edges()
print(net.nodes())
