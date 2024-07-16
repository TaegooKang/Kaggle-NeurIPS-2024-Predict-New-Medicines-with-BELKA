import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GATv2Conv

from .utils import F_unpackbits


class GAT(nn.Module):
    def __init__(self, hidden_dim, edge_dim, num_heads, num_layers, num_classes=3):
        super().__init__()

        self.num_layers = num_layers
        self.atom_emb = nn.Embedding(120, hidden_dim//2)
        self.atom_lin = nn.Linear(56, hidden_dim//2)
        self.edge_emb = nn.Embedding(22, edge_dim)
        self.edge_lin = nn.Linear(8, edge_dim)
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GATv2Conv(hidden_dim, hidden_dim//num_heads, num_heads, edge_dim=2*edge_dim, concat=True))
            else:
                self.convs.append(GATv2Conv(hidden_dim, hidden_dim, heads=1, edge_dim=2*edge_dim, concat=False))

        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.lin = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )


    def extract_feature(self, x, edge_index, edge_attr, batch):
        x = torch.cat([self.atom_emb(x[:,0].long()), self.atom_lin(F_unpackbits(x[:,1:], dim=-1).float())], dim=1)
        edge_feat = torch.cat([self.edge_emb(edge_attr[:,0].long()), self.edge_lin(F_unpackbits(edge_attr[:,1:], dim=-1).float())], dim=1)

        for layer in range(self.num_layers):
            if layer == self.num_layers - 1:
                x = self.norms[layer](self.convs[layer](x, edge_index, edge_feat))
            else:
                x = F.leaky_relu(self.norms[layer](self.convs[layer](x, edge_index, edge_feat)))
        
        x = global_mean_pool(x, batch)
        
        return x
    
    
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.extract_feature(x, edge_index, edge_attr, batch)
        bind = self.lin(x)
        
        return bind
    
    