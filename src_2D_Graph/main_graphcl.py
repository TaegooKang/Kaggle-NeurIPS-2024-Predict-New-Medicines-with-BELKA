

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import pandas as pd
import multiprocessing
import argparse
import time
import os 
import numpy as np 
import pickle 

from tqdm import tqdm 

from data_utils import drop_nodes, from_smiles
from utils import fix_seed, save_args, get_logger
from models.gin import GIN 
from models.mpnn import MPNN
from models.gat import GAT


def make_batch_graph(graph, index=None, device='cpu', aug_prob=0.0, aug_ratio=0.0):
    if index is None:
        index = np.arange(len(graph)).tolist()
    
    xs1, xs2 = [], []
    edge_indices1, edge_indices2 = [], []
    edge_attrs1, edge_attrs2 = [],[]
    batch1, batch2 = [], []
    offset1, offset2 = 0, 0
    for b, i in enumerate(index):
        x, edge_index, edge_attr = graph[i]
        
        # first view
        p = np.random.uniform(0, 1)
        if p < aug_prob:
            x1, edge_index1, edge_attr1 = drop_nodes(x, edge_index, edge_attr, aug_ratio)
            N = x1.shape[0]
            xs1.append(x1)
            edge_attrs1.append(np.concatenate([edge_attr1, edge_attr1], axis=0))
            edge_indices1.append(np.concatenate([edge_index1, edge_index1[:,[1,0]]], axis=0, dtype=int) + offset1)
            batch1 += N * [b]
            offset1 += N
        else:
            N = x.shape[0]
            xs1.append(x)
            edge_attrs1.append(np.concatenate([edge_attr, edge_attr], axis=0))
            edge_indices1.append(np.concatenate([edge_index, edge_index[:,[1,0]]], axis=0, dtype=int) + offset1)
            batch1 += N * [b]
            offset1 += N
        
        # second view
        
        p = np.random.uniform(0, 1)
        if p < aug_prob:
            x2, edge_index2, edge_attr2 = drop_nodes(x, edge_index, edge_attr, aug_ratio)
            N = x2.shape[0]
            xs2.append(x2)
            edge_attrs2.append(np.concatenate([edge_attr2, edge_attr2], axis=0))
            edge_indices2.append(np.concatenate([edge_index2, edge_index2[:,[1,0]]], axis=0, dtype=int) + offset2)
            batch2 += N * [b]
            offset2 += N
        else:
            N = x.shape[0]
            xs2.append(x)
            edge_attrs2.append(np.concatenate([edge_attr, edge_attr], axis=0))
            edge_indices2.append(np.concatenate([edge_index, edge_index[:,[1,0]]], axis=0, dtype=int) + offset2)
            batch2 += N * [b]
            offset2 += N
        
    xs1 = torch.from_numpy(np.concatenate(xs1)).to(device)
    edge_attrs1 = torch.from_numpy(np.concatenate(edge_attrs1, dtype=np.uint8)).to(device)
    edge_indices1 = torch.from_numpy(np.concatenate(edge_indices1).T).to(device)
    batch1 = torch.LongTensor(batch1).to(device)
    
    xs2 = torch.from_numpy(np.concatenate(xs2)).to(device)
    edge_attrs2 = torch.from_numpy(np.concatenate(edge_attrs2, dtype=np.uint8)).to(device)
    edge_indices2 = torch.from_numpy(np.concatenate(edge_indices2).T).to(device)
    batch2 = torch.LongTensor(batch2).to(device)
    
    return {'A': (xs1, edge_attrs1, edge_indices1, batch1),
            'B': (xs2, edge_attrs2, edge_indices2, batch2)}
    
    
def graph_loader(queue, graphs, batch_size, shuffle=False, drop_last=False, aug_prob=0.9, aug_ratio=0.2):
    # 83653268
    g_index = [i for i in range(len(graphs))]
    if shuffle: np.random.shuffle(g_index)
    for _, index in enumerate(np.arange(0, len(g_index), batch_size)):
        index = g_index[index:index+batch_size]
        if len(index)!= batch_size: 
            if drop_last: continue #drop last

        batch = {
			'graphs': make_batch_graph(graphs, index, device='cpu', aug_prob=aug_prob, aug_ratio=aug_ratio),
        }
        queue.put(batch)
    queue.put(None)
    

def load_graphs():
    path = '/data/datasets/leash-BELKA/random_stratified_split/bb_enc.pickle'
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    smiles = []
    for _, v in data.items():
        smiles += v
    smiles = list(set(smiles))
    
    print("Generate building block graphs..") 
    graphs = []
    for s in tqdm(smiles):
        graphs.append(from_smiles(s))
    print("# number of graphs:", len(graphs))
    
    return graphs

class GraphCL(nn.Module):
    def __init__(self, gnn, sim_metric, T):
        super(GraphCL, self).__init__()
        self.gnn = gnn
        self.sim_metric = sim_metric
        self.T = T
        self.projection_head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Linear(128, 128))

    def forward_cl(self, x, edge_index, edge_attr, batch):
        x = self.gnn.extract_feature(x, edge_index, edge_attr, batch)
        x = self.projection_head(x)
        return x

    def loss_cl(self, x1, x2):
        
        loss, acc = dual_CL(x1, x2, self.sim_metric, self.T)
        
        return loss, acc
    

def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


def do_CL(X, Y, sim_metric, T=0.1, neg_samples=1):
    
    X = F.normalize(X, dim=-1)
    Y = F.normalize(Y, dim=-1)

    if sim_metric == 'InfoNCE_dot_prod':
        criterion = nn.CrossEntropyLoss()
        B = X.size()[0]
        logits = torch.mm(X, Y.transpose(1, 0))  # B*B
        logits = torch.div(logits, T)
        labels = torch.arange(B).long().to(logits.device)  # B*1

        CL_loss = criterion(logits, labels)
        pred = logits.argmax(dim=1, keepdim=False)
        CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B

    elif sim_metric == 'EBM_dot_prod':
        criterion = nn.BCEWithLogitsLoss()
        neg_Y = torch.cat([Y[cycle_index(len(Y), i + 1)]
                           for i in range(neg_samples)], dim=0)
        neg_X = X.repeat((neg_samples, 1))

        pred_pos = torch.sum(X * Y, dim=1) / T
        pred_neg = torch.sum(neg_X * neg_Y, dim=1) / T

        loss_pos = criterion(pred_pos, torch.ones(len(pred_pos)).to(pred_pos.device))
        loss_neg = criterion(pred_neg, torch.zeros(len(pred_neg)).to(pred_neg.device))
        CL_loss = loss_pos + neg_samples * loss_neg

        CL_acc = (torch.sum(pred_pos > 0).float() +
                  torch.sum(pred_neg < 0).float()) / \
                 (len(pred_pos) + len(pred_neg))
        CL_acc = CL_acc.detach().cpu().item()

    else:
        raise Exception

    return CL_loss, CL_acc


def dual_CL(X, Y, sim_metric, T):
    CL_loss_1, CL_acc_1 = do_CL(X, Y, sim_metric, T)
    CL_loss_2, CL_acc_2 = do_CL(Y, X, sim_metric, T)
    return (CL_loss_1 + CL_loss_2) / 2, (CL_acc_1 + CL_acc_2) / 2
    
    
def get_args_parser():
    parser = argparse.ArgumentParser('[Kaggle] Leash-BELKA Competition Training', add_help=False)
    parser.add_argument('--exp-name', '-ex', default='exp', type=str)
    
    # seed
    parser.add_argument('--seed', default=10, type=int)
    
    # training
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    
    # learning rate 
    parser.add_argument('--base-lr', default=1e-3, type=float)
  
    # optimizer
    parser.add_argument('--weight-decay', '-wd', default=5e-2, type=float)
    
    # logging
    parser.add_argument('--log-dir', default='/data/wlsghldud/ktg/kaggle/checkpoints', type=str)
    
    # model config
    parser.add_argument('--hidden-channels', default=64, type=int)
    parser.add_argument('--num-layers', default=4, type=int)
    parser.add_argument('--pool', default='mean', choices=['mean', 'sum'])
    parser.add_argument('--sim-metric', default='InfoNCE_dot_prod', choices=['InfoNCE_dot_prod', 'EBM_dot_prod'])
    parser.add_argument('--T', default=0.3, type=float)
    
    # aug config
    parser.add_argument('--aug-prob', default=1.0, type=float)
    parser.add_argument('--aug-ratio', default=0.3, type=float)
    
    return parser

if __name__ == '__main__':
    
    parser = get_args_parser()
    args = parser.parse_args()
    fix_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    
   # Create logging directory
    log_dir = os.path.join(args.log_dir, args.exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Save arguments
    save_args(args, log_dir)
    
    # Create train logger
    logger = get_logger(name='train',
                        file_path=os.path.join(log_dir, 'train.log'),
                        stream=True,
                        level='info')
    
    # Prepare model
    logger.info('Build model...')
    gnn = GIN(hidden_channels=args.hidden_channels, num_layers=args.num_layers, pool=args.pool)
    #gnn = MPNN(64, 64, 4)
    #gnn = GAT(128, 16, 8, 4)
    model = GraphCL(gnn, args.sim_metric, args.T).to(device)
    
    # Prepare optimizer ======================================================== #
    logger.info('Build optimizer...')
    optimizer = optim.AdamW(model.parameters(), 
                            betas=(0.9, 0.95), 
                            weight_decay=args.weight_decay, 
                            lr=args.base_lr)
    # ========================================================================== #
    
    # Prepare graph datasets ================================================= #
    # load graph datasets
    graphs = load_graphs()
    logger.info(f'# of train graphs: {len(graphs)}')
    # ========================================================================== #
    
    # amp scaler
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Training procedure
    for epoch in range(args.epochs):
        
        train_queue = multiprocessing.Manager().Queue(maxsize=32)
        train_loader_process = multiprocessing.Process(target=graph_loader, 
                                                       args=(train_queue, graphs, args.batch_size, True, True, args.aug_prob, args.aug_ratio))
        train_loader_process.start()
    
        model.train()
        train_loss = 0
        train_acc = 0
        # drop last
        len_train_loader = len(graphs) // args.batch_size
        
        for _ in range(len_train_loader):
            
            batch_graphs = train_queue.get()
            if batch_graphs is None:
                break
            
            with torch.cuda.amp.autocast():
                xs, edge_attrs, edge_indices, batch = batch_graphs['graphs']['A']
                xs = xs.to(device)
                edge_indices = edge_indices.to(device)
                edge_attrs = edge_attrs.to(device)
                batch = batch.to(device)
                z1 = model.forward_cl(xs, edge_indices, edge_attrs, batch)

                xs, edge_attrs, edge_indices, batch = batch_graphs['graphs']['B']
                xs = xs.to(device)
                edge_indices = edge_indices.to(device)
                edge_attrs = edge_attrs.to(device)
                batch = batch.to(device)
                z2 = model.forward_cl(xs, edge_indices, edge_attrs, batch)
                
                loss, acc = model.loss_cl(z1, z2)
          
            
            # backward
            optimizer.zero_grad()
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward() 
                scaler.step(optimizer)
                scaler.update()
                torch.clear_autocast_cache()
        
            # logging
            train_loss += loss.detach().item()
            train_acc += acc
            
        train_loader_process.join()
        train_loss /= len_train_loader
        train_acc /= len_train_loader
        logger.info(f'[Epoch {epoch+1}/{args.epochs}] Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.4f}')

    torch.save({'model': model.gnn.state_dict()}, os.path.join(log_dir, 'gnn.pth'))