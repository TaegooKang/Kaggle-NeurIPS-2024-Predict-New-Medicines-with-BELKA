import os  
import numpy as np
import pandas as pd 

import torch
from torch.utils.data import Dataset


def get_filenames(directory, ext):
    filenames = []
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(ext):
                g_idx = int(entry.name.split('.')[0][5:])
                filenames.append(g_idx)
    return filenames


class Graph3dDataset(Dataset):
    def __init__(self, graph_paths, targets): 
        self.graph_paths = graph_paths
        self.targets = targets
        
    def __getitem__(self, index):
        g_idx = self.graph_paths[index]
        graph = np.load(f'/data2/local_datasets/leash-BELKA/3d-graphs/train/graph{g_idx}.npy')
        y = self.targets[g_idx]
        return graph[:,0], graph[:,1:], y
        
    def __len__(self):
        return len(self.graph_paths)


class Graph3dTestDataset(Dataset):
    def __init__(self): 
        self.name = 'test dataset'
        
    def __getitem__(self, index):
        
        graph = np.load(f'/data2/local_datasets/leash-BELKA/3d-graphs/test/graph{index}.npy')
        return graph[:,0], graph[:,1:]
        
    def __len__(self):
        return 1674896
    
    
def build_3dgraph_dataset():
    directory = '/data2/local_datasets/leash-BELKA/3d-graphs/train'
    ext = '.npy'
    graph_paths = get_filenames(directory, ext)
    
    num_train = 94000000
    train_graph_paths = graph_paths[:num_train]
    valid_graph_paths = graph_paths[num_train:]
    
    targets_0 = np.load('/data/datasets/leash-BELKA/random_stratified_split/train_target.npy')
    targets_1 = np.load('/data/datasets/leash-BELKA/random_stratified_split/valid_target.npy')
    targets = np.concatenate([targets_0, targets_1], axis=0)
    # print(targets.shape)
    
    trainset = Graph3dDataset(train_graph_paths, targets)
    validset = Graph3dDataset(valid_graph_paths, targets)
    
    return trainset, validset


def my_collate_fn_3d(batch_data):
    z = []
    pos = []
    batch = []
    y = []
    offset = 0
    for b, data in enumerate(batch_data):
        if len(data) == 2:
            _z, _pos = data
        elif len(data) == 3:
            _z, _pos, _y = data
            y.append(_y)
        N = _z.shape[0]
        z.append(_z)
        pos.append(_pos)
        batch += N * [b]
        offset += N
        
    z = torch.from_numpy(np.concatenate(z, axis=0)).long()
    pos = torch.from_numpy(np.concatenate(pos, axis=0))
    batch = torch.LongTensor(batch)
    
    if len(y) == 0:
        return z, pos, batch
    else:
        y = torch.from_numpy(np.stack(y)).float()
        return z, pos, batch, y