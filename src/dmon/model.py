import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init


class InputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InputLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):
    def __init__(self, in_channels, n_clusters, dropout):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, n_clusters),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return self.mlp(x)


def _weights_init(m):
    if isinstance(m, nn.Linear):
        init.orthogonal_(m.weight)
        init.constant_(m.bias, 0)


class DMON(nn.Module):
    def __init__(self, model_lists, in_channels, n_clusters, adjacency, lr, device,
                 collapse_regularization=0.1, dropout_rate=0,):
        super(DMON, self).__init__()
        self.n_clusters = n_clusters
        self.adjacency = adjacency
        self.device = device
        self.collapse_regularization = collapse_regularization
        self.dropout_rate = dropout_rate
        self.transform = MLP(in_channels, n_clusters, dropout_rate).to(device)
        self.transform.apply(_weights_init)
        parameters_list = list()
        for model in model_lists:
            parameters_list += list(model.parameters())
        self.optimizer = optim.Adam(parameters_list + list(self.transform.parameters()), lr=lr)

    def _loss(self, u_features, v_features, epoch):
        adjacency = self.adjacency.cuda()
        u_assignments = torch.softmax(self.transform(u_features), dim=1)
        v_assignments = torch.softmax(self.transform(v_features), dim=1)
        u_cluster_sizes = torch.sum(u_assignments, dim=0)
        v_cluster_sizes = torch.sum(v_assignments, dim=0)
        u_degrees = torch.sum(self.adjacency, dim=1).view(-1, 1)
        v_degrees = torch.sum(self.adjacency, dim=0).view(1, -1)
        u_number_of_nodes = adjacency.shape[0]
        v_number_of_nodes = adjacency.shape[1]
        number_of_edges = torch.sum(u_degrees)

        # U^T * A * V
        graph_pooled = torch.mm(u_assignments.t(), adjacency)
        graph_pooled = torch.mm(graph_pooled, v_assignments)

        # U^T * d^T * g * V
        normalizer_left = torch.mm(u_assignments.t(), u_degrees)
        normalizer_right = torch.mm(v_degrees, v_assignments)
        normalizer = torch.mm(normalizer_left, normalizer_right) / number_of_edges

        spectral_loss = -torch.trace(graph_pooled - normalizer) / number_of_edges
        # u_collapse_loss = torch.norm(u_cluster_sizes) / u_number_of_nodes * float(self.n_clusters) ** 0.5 - 1
        # v_collapse_loss = torch.norm(v_cluster_sizes) / v_number_of_nodes * float(self.n_clusters) ** 0.5 - 1
        u_collapse_loss = torch.norm(torch.mm(u_assignments.t(), u_assignments)/torch.norm(
            torch.mm(u_assignments.t(), u_assignments))-torch.eye(self.n_clusters).cuda()/float(self.n_clusters) ** 0.5)
        v_collapse_loss = torch.norm(torch.mm(v_assignments.t(), v_assignments) / torch.norm(
            torch.mm(v_assignments.t(), v_assignments)) - torch.eye(self.n_clusters).cuda() / float(self.n_clusters) ** 0.5)
        loss = spectral_loss + self.collapse_regularization * (u_collapse_loss + v_collapse_loss)
        print('epoch: %s, spectral_loss=%.7s, u_collapse_loss=%.7s, v_collapse_loss=%.7s, loss=%.10s'
              % (epoch, spectral_loss.item(), u_collapse_loss.item(), v_collapse_loss.item(), loss.item()))
        return loss

    def forward(self, u_inputs, v_inputs):
        u_out = torch.softmax(self.transform(u_inputs), dim=1)
        v_out = torch.softmax(self.transform(v_inputs), dim=1)
        return u_out, v_out

    def forward_backward(self, u_inputs, v_inputs, epoch):
        loss = self._loss(u_inputs, v_inputs, epoch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




