import numpy as np 

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, GINEConv

from .utils import F_unpackbits

pool = {
    'mean': global_mean_pool,
    'sum': global_add_pool
}

class GIN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers=4, num_classes=3, pool='mean'):
        super().__init__()

        self.num_layers = num_layers
        self.atom_emb = nn.Embedding(120, hidden_channels)
        self.atom_lin = nn.Linear(56, hidden_channels)
        self.edge_emb = nn.Embedding(22, hidden_channels)
        self.edge_lin = nn.Linear(8, hidden_channels)
        self.pool = pool
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            l = nn.Sequential(
                nn.Linear(2 * hidden_channels, 4 * hidden_channels),
                nn.BatchNorm1d(4 * hidden_channels),
                nn.ReLU(),
                nn.Linear(4 * hidden_channels, 2 * hidden_channels),
                nn.BatchNorm1d(2 * hidden_channels)
            )
            self.convs.append(GINEConv(l, train_eps=False))

       
        self.lin = nn.Sequential(
            nn.Linear(2 * hidden_channels, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes) # 0.435 모델은 bias 있음.
        )

    def extract_feature(self, x, edge_index, edge_attr, batch):
        x = torch.cat([self.atom_emb(x[:,0].long()), self.atom_lin(F_unpackbits(x[:,1:], dim=-1).float())], dim=1)
        edge_feat = torch.cat([self.edge_emb(edge_attr[:,0].long()), self.edge_lin(F_unpackbits(edge_attr[:,1:], dim=-1).float())], dim=1)

        for layer in range(self.num_layers):
            if layer == self.num_layers - 1:
                x = self.convs[layer](x, edge_index, edge_feat)
            else:
                x = F.relu(self.convs[layer](x, edge_index, edge_feat)) 

        x = pool[self.pool](x, batch)
        
        return x
    
    def forward(self, x, edge_index, edge_attr, batch):
        
        x = self.extract_feature(x, edge_index, edge_attr, batch)
        x = self.lin(x)
        return x
    
 