import torch
import torch_geometric
import torch.nn.functional as F


class GraphSAGE(torch.nn.Module):
    name = 'GraphSAGE'

    def __init__(self):
        super(GraphSAGE, self).__init__()
        self.conv1 = torch_geometric.nn.SAGEConv((-1, -1), 16)
        self.conv2 = torch_geometric.nn.SAGEConv((-1, -1), 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GIN(torch.nn.Module):  # Good for graphs without node features
    name = 'GIN'

    def __init__(self, in_channels, out_channels):
        super(GIN, self).__init__()
        self.conv1 = torch_geometric.nn.GINConv(torch.nn.Sequential(torch.nn.Linear(in_channels, 16), torch.nn.ReLU(),
                                                                    torch.nn.Linear(16, out_channels)))
        self.conv2 = torch_geometric.nn.GINConv(torch.nn.Sequential(torch.nn.Linear(out_channels, 16), torch.nn.ReLU(),
                                                                    torch.nn.Linear(16, out_channels)))

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class GCN(torch.nn.Module):
    name = 'GCN'

    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = torch_geometric.nn.GCNConv(in_channels, 16)
        self.conv2 = torch_geometric.nn.GCNConv(16, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class GAT(torch.nn.Module):
    name = 'GAT'

    def __init__(self):
        super(GAT, self).__init__()
        self.conv1 = torch_geometric.nn.GATConv((-1, -1), 16, add_self_loops=False)
        self.conv2 = torch_geometric.nn.GATConv(16, 1, add_self_loops=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
