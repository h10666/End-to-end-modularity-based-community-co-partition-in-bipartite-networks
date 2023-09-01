import torch
import torch.nn as nn
from src.gcn.layer import GraphConvolution


class GCN(nn.Module):
    def __init__(self, infeat, hidfeat, outfeat):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(infeat, hidfeat)
        self.gc2 = GraphConvolution(hidfeat, outfeat)
        # self.gc3 = GraphConvolution(hidfeat2, outfeat)

    def forward(self, x, adj):
        x = torch.selu(self.gc1(x, adj))
        x = torch.dropout(x, p=0.4, train=self.training)
        x = torch.selu(self.gc2(x, adj))
        # x = torch.dropout(x, p=0.4, train=self.training)
        # x = torch.selu(self.gc3(x, adj))
        return x
