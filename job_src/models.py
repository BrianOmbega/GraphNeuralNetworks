import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINConv
from torch_geometric.nn import SAGEConv

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=1):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x





class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x




class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv1 = GINConv(mlp)
        self.conv2 = GINConv(mlp)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x




class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x




class EdgeClassifier(torch.nn.Module):
    def __init__(self, emb_dim, edge_feat_dim):
        super(EdgeClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(2 * emb_dim + edge_feat_dim, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, z, edge_index, edge_attr):
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        x = torch.cat([src, dst, edge_attr], dim=1)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).squeeze()
