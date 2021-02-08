import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


# Simple Graph Convolution Network
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.sigmoid()
        return x


# Graph Attention Network
class GATN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim, num_heads):
        super(GATN, self).__init__()
        torch.manual_seed(12345)
        self.gat1 = GATConv(in_channels=num_features,
                            out_channels=hidden_dim,
                            heads=num_heads,
                            concat=True,
                            dropout=0.2,
                            add_self_loops=True,
                            bias=True)
        self.gat2 = GATConv(in_channels=hidden_dim * num_heads,
                            out_channels=num_classes,
                            heads=1,
                            concat=True,
                            dropout=0.0,
                            add_self_loops=True,
                            bias=True)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = x.relu()
        x = self.gat2(x, edge_index)
        x = x.sigmoid()
        return x
