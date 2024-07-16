import torch
import torch.nn as nn 
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_scatter import scatter

from .utils import F_unpackbits



class MPNNLayer(MessagePassing):
    def __init__(self, emb_dim=64, edge_dim=4, aggr='add'):
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim
        self.mlp_msg = nn.Sequential(
            nn.Linear(2 * emb_dim + edge_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim)
        )
        self.mlp_upd = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim)
        )

    def forward(self, h, edge_index, edge_attr):
        out = self.propagate(edge_index, h=h, edge_attr=edge_attr)
        return out

    def message(self, h_i, h_j, edge_attr):
        msg = torch.cat([h_i, h_j, edge_attr], dim=-1)
        return self.mlp_msg(msg)

    def aggregate(self, inputs, index):
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)

    def update(self, aggr_out, h):
        upd_out = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_upd(upd_out)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')


class MPNNEncoder(nn.Module):
    def __init__(self, emb_dim=64, edge_dim=4, num_layers=4):
        super().__init__()

        self.num_layers = num_layers
        # Stack of MPNN layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(MPNNLayer(emb_dim, edge_dim, aggr='add'))

        self.pool = global_mean_pool

    def forward(self, x, edge_index, edge_attr, batch): #PyG.Data - batch of PyG graphs

        h = x.float()

        for layer in range(self.num_layers):
            if layer == self.num_layers - 1:
                h = h + self.convs[layer](h, edge_index.long(), edge_attr.float())  # (n, d) -> (n, d)
            else:
                h = h + F.relu(self.convs[layer](h, edge_index.long(), edge_attr.float()))
                
        h_graph = self.pool(h, batch)  
        return h_graph


class MPNN(nn.Module):
    def __init__(self, atom_dim, edge_dim, num_layers, num_classes=3):
        super().__init__()
 
        self.atom_emb = nn.Embedding(120, atom_dim)
        self.atom_lin = nn.Linear(56, atom_dim)
        self.edge_emb = nn.Embedding(22, edge_dim)
        self.edge_lin = nn.Linear(8, edge_dim)
        self.smile_encoder = MPNNEncoder(
            emb_dim=2*atom_dim, edge_dim=2*edge_dim, num_layers=num_layers,
        )
        self.classifier = nn.Sequential(
            nn.Linear(2*atom_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x, edge_index, edge_attr, batch):
        atom_feat = torch.cat([self.atom_emb(x[:,0].long()), self.atom_lin(F_unpackbits(x[:,1:], dim=-1).float())], dim=1)
        edge_feat = torch.cat([self.edge_emb(edge_attr[:,0].long()), self.edge_lin(F_unpackbits(edge_attr[:,1:], dim=-1).float())], dim=1)

        x = self.smile_encoder(atom_feat, edge_index, edge_feat, batch) 
        
        bind = self.classifier(x)

        return bind
    
    
    