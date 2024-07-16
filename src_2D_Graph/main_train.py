import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import pandas as pd
import multiprocessing
import argparse
import time
import os

from tqdm import tqdm

from models.gin import GIN
from models.gat import GAT 
from models.mpnn import MPNN
from data_utils import *
from utils import format_time, fix_seed, ap, save_args, get_logger


def get_args_parser():
    parser = argparse.ArgumentParser('[Kaggle] Leash-BELKA Competition Training', add_help=False)
    parser.add_argument('--exp-name', '-ex', default='exp', type=str)
    
    # seed
    parser.add_argument('--seed', default=10, type=int)
    
    # training
    parser.add_argument('--batch-size', '-bs', default=4096, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    
    # learning rate 
    parser.add_argument('--base-lr', '-lr', default=1e-3, type=float)
    parser.add_argument('--decay-step1', '-ds1', default=10, type=int)
    parser.add_argument('--decay-step2', '-ds2', default=15, type=int)
    
    # optimizer
    parser.add_argument('--weight-decay', '-wd', default=5e-2, type=float)
    
    # logging
    parser.add_argument('--log-dir', default='./checkpoints', type=str)
    
    # model config
    parser.add_argument('--model', choices=['MPNN', 'GAT', 'GIN'])
    parser.add_argument('--pretrained', typd=str)
    
    # with Hydrogen
    parser.add_argument('--with-H', default=False, action='store_true')
    
    # use full dataset
    parser.add_argument('--full', default=False, action='store_true')
    
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
    if args.model == 'MPNN':
        model = MPNN(atom_dim=64, edge_dim=64, num_layers=4).to(device)
    elif args.model == 'GIN':
        model = GIN(hidden_channels=128, num_layers=4).to(device)
    elif args.model == 'GAT':
        model = GAT(hidden_dim=128, edge_dim=16, num_heads=8, num_layers=4).to(device)
    
    # Load pretrained checkpoint
    if args.pretrained is not None:
        logger.info('Pretrained ckpt: '+args.pretrained)
        ckpt = torch.load(args.pretrained, map_location='cpu')['model']
        msg = model.load_state_dict(ckpt, strict=False)
        
        missing_keys = 'missing keys: '
        if len(msg.missing_keys) == 0:
            missing_keys += 'None'
        else:
            for k in msg.missing_keys:
                missing_keys += str(k) + ', '
        
        unexpected_keys = 'unexpected keys: '
        if len(msg.unexpected_keys) == 0:
            unexpected_keys += 'None'
        else:
            for k in msg.unexpected_keys:
                unexpected_keys += str(k) + ', '
            
        logger.info(missing_keys)
        logger.info(unexpected_keys)
    
    # Prepare optimizer ======================================================== #
    logger.info('Build optimizer...')
    optimizer = optim.AdamW(model.parameters(), 
                            betas=(0.9, 0.999), 
                            weight_decay=args.weight_decay, 
                            lr=args.base_lr)
    # ========================================================================== #
    
    # Prepare graph datasets ================================================= #
    logger.info('Build dataset..')
    s = time.time()
    if args.with_H:
        train_graphs = load_compressed_ibz2_pickle('./leash-BELKA/random_stratified_split/train-graphs-wH.pickle.01.b2z') \
                        + load_compressed_ibz2_pickle('./leash-BELKA/random_stratified_split/train-graphs-wH.pickle.02.b2z') \
                        + load_compressed_ibz2_pickle('./leash-BELKA/random_stratified_split/train-graphs-wH.pickle.03.b2z')
        train_targets = np.load('./leash-BELKA/random_stratified_split/train_target.npy') 
        valid_graphs = load_compressed_ibz2_pickle('./leash-BELKA/random_stratified_split/valid-graphs-wH.pickle.b2z')
        valid_targets = np.load('./leash-BELKA/random_stratified_split/valid_target.npy') 
    else:
        train_graphs = load_compressed_ibz2_pickle('./leash-BELKA/random_stratified_split/train-graphs-30m.pickle.01.b2z') \
                        + load_compressed_ibz2_pickle('./leash-BELKA/random_stratified_split/train-graphs-30m.pickle.02.b2z') \
                        + load_compressed_ibz2_pickle('./leash-BELKA/random_stratified_split/train-graphs-30m.pickle.03.b2z')
        train_targets = np.load('./leash-BELKA/random_stratified_split/train_target.npy') 
        valid_graphs = load_compressed_ibz2_pickle('./leash-BELKA/random_stratified_split/valid-graphs.pickle.b2z')
        valid_targets = np.load('./leash-BELKA/random_stratified_split/valid_target.npy') 
    
    if args.full:
        train_graphs += valid_graphs
        train_targets = np.concatenate((train_targets, valid_targets), axis=0)
        e = time.time()
        logger.info(f'# of train graphs: {len(train_graphs)}')
        logger.info(f'Finished!! Dataset loading time: {format_time(e-s)}..')
    
    else:
        e = time.time()
        logger.info(f'# of train graphs: {len(train_graphs)}')
        logger.info(f'# of valid graphs: {len(valid_graphs)}')
        logger.info(f'Finished!! Dataset loading time: {format_time(e-s)}..')
    # ========================================================================== #
    
    # loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # amp scaler
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    decay_step = [args.decay_step1, args.decay_step2]
    
    # Training procedure
    for epoch in range(args.epochs):
        train_queue = multiprocessing.Manager().Queue(maxsize=32)
        train_loader_process = multiprocessing.Process(target=graph_loader, 
                                                       args=(train_queue, train_graphs, train_targets, args.batch_size, True, True))
        train_loader_process.start()
    
        model.train()
        train_loss = 0
        # drop last
        len_train_loader = len(train_graphs) // args.batch_size
        
        for _ in tqdm(range(len_train_loader)):
            
            batch_graphs = train_queue.get()
            if batch_graphs is None:
                break
            xs, edge_indices, edge_attrs, batch = batch_graphs['graphs']
            xs = xs.to(device)
            edge_indices = edge_indices.to(device)
            edge_attrs = edge_attrs.to(device)
            batch = batch.to(device)
            targets = batch_graphs['targets'].to(device)
            # forward 
            with torch.cuda.amp.autocast():
                y_hat = model(xs, edge_indices, edge_attrs, batch)
                loss = criterion(y_hat, targets)
                
                
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
        
        train_loader_process.join()
        train_loss /= len_train_loader
        logger.info(f'[Epoch {epoch+1}/{args.epochs}] Train Loss: {train_loss:.5f}')
        
        # multistep lr schedule
        if epoch + 1 in decay_step:
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                param_group['lr'] = lr * 0.1
        
        if not args.full:
            ## validation
            valid_queue = multiprocessing.Manager().Queue(maxsize=32)
            valid_loader_process = multiprocessing.Process(target=graph_loader, args=(valid_queue, valid_graphs, valid_targets, args.batch_size))
            valid_loader_process.start()
            
            model.eval()
            val_loss = 0
            preds = []
            ys = []
            stat =  '-----------------------------------------------------\n'
            stat += '|       |   BRD4   |   HSA   |   sEH   |    loss    |\n'
            stat += '-----------------------------------------------------'
            
            len_valid_loader = (len(valid_graphs) // args.batch_size) + 1
            with torch.no_grad():
                for _ in tqdm(range(len_valid_loader)):
                    batch_graphs = valid_queue.get()
                    xs, edge_indices, edge_attrs, batch = batch_graphs['graphs']
                    xs = xs.to(device)
                    edge_indices = edge_indices.to(device)
                    edge_attrs = edge_attrs.to(device)
                    batch = batch.to(device)
                    targets = batch_graphs['targets'].to(device)
        
                    # forward 
                    with torch.cuda.amp.autocast():
                        y_hat = model(xs, edge_indices, edge_attrs, batch)
                        loss = criterion(y_hat, targets)
                        pred = torch.sigmoid(y_hat).cpu().numpy()
                    
                    val_loss += loss.item()
                    
                    preds.append(pred)
                    ys.append(targets.cpu().numpy())
                    
            preds = np.concatenate(preds, axis=0)
            ys = np.concatenate(ys, axis=0)
            
            ap_BRD4 = ap(preds[:,0], ys[:,0])
            ap_HSA = ap(preds[:,1], ys[:,1])
            ap_sEH = ap(preds[:,2], ys[:,2])
            
            val_loss /= len_valid_loader
            stat += f'\n| share |   {ap_BRD4:.3f}  |  {ap_HSA:.3f}  |  {ap_sEH:.3f}  |  {val_loss:.6f}  |'
            stat += '\n-----------------------------------------------------'
            logger.info(f'[Epoch {epoch+1}/{args.epochs}] Validation Stats.. \n' + stat)

        if epoch+1 >= 10:
            torch.save({'model': model.state_dict()}, os.path.join(log_dir, f'model_{epoch+1}epoch.pth'))
    
    # Inference
    print('Start Inference!!')
    print('Build test graphs...')
    submit = pd.read_parquet('./leash-BELKA/origin/test.parquet')
    test_smiles = submit['molecule_smiles'].values

    with multiprocessing.Pool(processes=32) as pool:
        test_graphs = list(tqdm(pool.imap(from_smiles, test_smiles), total=len(test_smiles)))
    
    print('# of test graphs:', len(test_graphs))
    test_index = [i for i in range(len(test_graphs))]
    len_test_loader = (len(test_graphs) // args.batch_size) + 1
    model.eval()
    preds = []
    with torch.no_grad():
        for _, index in tqdm(enumerate(np.arange(0, len(test_index), args.batch_size)), total=len_test_loader):
            index = test_index[index:index+args.batch_size]
            batch_graphs = make_batch_graph(test_graphs, index)
            xs, edge_indices, edge_attrs, batch = batch_graphs
            xs = xs.to(device)
            edge_indices = edge_indices.to(device)
            edge_attrs = edge_attrs.to(device)
            batch = batch.to(device)
            with torch.cuda.amp.autocast():
                z = model(xs, edge_indices, edge_attrs, batch)
                z = torch.sigmoid(z).cpu().numpy()
                preds.append(z)
    
    preds = np.concatenate(preds, 0)
    
    print(preds.shape)    
    submit['binds'] = 0
    submit.loc[submit['protein_name']=='BRD4', 'binds'] = preds[(submit['protein_name']=='BRD4').values, 0]
    submit.loc[submit['protein_name']=='HSA', 'binds'] = preds[(submit['protein_name']=='HSA').values, 1]
    submit.loc[submit['protein_name']=='sEH', 'binds'] = preds[(submit['protein_name']=='sEH').values, 2]
    submit[['id', 'binds']].to_csv(os.path.join(log_dir, 'submission.csv'), index = False)
    
    