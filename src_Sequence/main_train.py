import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader

import pandas as pd
import numpy as np 
import argparse
import time
import os

from tqdm import tqdm

from models.mamba import MambaLM
from models.cnn import CNN

from dataset import SequenceDataset
from utils import format_time, fix_seed, ap, save_args, get_logger

    
def get_args_parser():
    parser = argparse.ArgumentParser('[Kaggle] Leash-BELKA Competition Training', add_help=False)
    
    parser.add_argument( '--exp-name', '-ex', default='exp', type=str)
    # seed
    parser.add_argument('--seed', default=10, type=int)
    
    # training
    parser.add_argument('--batch-size', default=4096, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    
    # optimizer
    parser.add_argument('--weight-decay', default=5e-2, type=float)
    parser.add_argument('--base-lr', default=1e-3, type=float)
    parser.add_argument('--lr-decay-factor', default=0.1, type=float)
    
    # logging
    parser.add_argument('--log-dir', default='./checkpoints', type=str)
    
    # dataset
    parser.add_argument('--tokenize', default='ch', type=str, choices=['ch', 'ais'])
    
    # model config
    parser.add_argument('--model', choices=['CNN', 'Mamba'])
    parser.add_argument('--emb-dim', default=96, type=int)
    parser.add_argument('--hidden-dim', default=96, type=int)
    parser.add_argument('--num-layers', default=4, type=int)
    
    return parser


if __name__ == "__main__":
    
    parser = get_args_parser()
    args = parser.parse_args()
    fix_seed(args.seed)
    
    # cudnn.deterministic = True
    cudnn.benchmark = True
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    
    NUM_EMB = {
        'ch': 37, # 0: padding, 1-36: tokens
        'ais': 222  # 0: padding, 1-220: tokens, 221: UNK
    }
    # Prepare model
    logger.info('Build model...')
    
    if args.model == 'CNN':
        model = CNN(num_embeddings=NUM_EMB[args.tokenize],
                    emb_dim=args.emb_dim,
                    hidden_dim=args.hiddden_dim,
                    num_classes=3)
    elif args.model == 'Mamba':
        model = MambaLM(num_embeddings=NUM_EMB[args.tokenize],
                emb_dim=args.emb_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                num_classes=3).to(device)
    model = nn.DataParallel(model)
    
    
    # # Prepare optimizer ======================================================== #
    logger.info('Build optimizer...')
    optimizer = optim.AdamW(model.parameters(), 
                            betas=(0.9, 0.999), 
                            weight_decay=args.weight_decay, 
                            lr=args.base_lr)
    # ========================================================================== #
    
   
    
    # Prepare graph datasets ================================================= #
    logger.info('Build dataset..')
    s = time.time()
    df_train = pd.read_parquet(f'/data/datasets/leash-BELKA/random_stratified_split/train_{args.tokenize}_tokenized.parquet')
    df_valid = pd.read_parquet(f'/data/datasets/leash-BELKA/random_stratified_split/valid_{args.tokenize}_tokenized.parquet')
    
    trainset = SequenceDataset(df_train)
    validset = SequenceDataset(df_valid)
    
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16, drop_last=True)
    valid_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False, num_workers=16, drop_last=False)
    
    e = time.time()
    logger.info(f'# of train smiles: {len(trainset)}')
    logger.info(f'# of valid smiles: {len(validset)}')
    logger.info(f'Finished!! Dataset loading time: {format_time(e-s)}..')
    # ========================================================================== #
    
    # loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # amp scaler
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    decay_step = [10, 15]
    
    # Training procedure
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        for batch_data in tqdm(train_loader):
            
            x, mask, y = batch_data
            x = x.to(device)
            mask = mask.to(device)
            y = y.to(device)
            
            # forward 
            with torch.cuda.amp.autocast():
                y_hat = model(x, mask)
                loss = criterion(y_hat, y)
        
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
        
        train_loss /= len(train_loader)
        logger.info(f'[Epoch {epoch+1}/{args.epochs}] Train Loss: {train_loss:.5f}')
        
        # multistep lr schedule
        if epoch + 1 in decay_step:
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                param_group['lr'] = lr * args.lr_decay_factor
                
        ## validation
        model.eval()
        val_loss = 0
        preds = []
        ys = []
        stat =  '-----------------------------------------------------\n'
        stat += '|       |   BRD4   |   HSA   |   sEH   |    loss    |\n'
        stat += '-----------------------------------------------------'
        
        with torch.no_grad():
            for batch_data in tqdm(valid_loader):
                x, mask, y = batch_data
                x = x.to(device)
                mask = mask.to(device)
                y = y.to(device)
    
                # forward 
                with torch.cuda.amp.autocast():
                    y_hat = model(x, mask)
                    loss = criterion(y_hat, y)
                    pred = torch.sigmoid(y_hat).cpu().numpy()
                
                val_loss += loss.item()
                
                preds.append(pred)
                ys.append(y.cpu().numpy())
                
        preds = np.concatenate(preds, axis=0)
        ys = np.concatenate(ys, axis=0)
        
        ap_BRD4 = ap(preds[:,0], ys[:,0])
        ap_HSA = ap(preds[:,1], ys[:,1])
        ap_sEH = ap(preds[:,2], ys[:,2])
        
        val_loss /= len(valid_loader)
        stat += f'\n| share |   {ap_BRD4:.3f}  |  {ap_HSA:.3f}  |  {ap_sEH:.3f}  |  {val_loss:.6f}  |'
        stat += '\n-----------------------------------------------------'
        logger.info(f'[Epoch {epoch+1}/{args.epochs}] Validation Stats.. \n' + stat)
    
    torch.save({'model': model.state_dict()}, os.path.join(log_dir, 'model.pth'))
    
    # Inference
    print('Start Inference!!')
    print('Build test smiles...')
    submit = pd.read_parquet('/data/datasets/leash-BELKA/origin/test.parquet')
    df_test = pd.read_parquet(f'/data/datasets/leash-BELKA/test_{args.tokenize}_tokenized.parquet')
    #df_test = pd.read_parquet('/data/datasets/leash-BELKA/test_ais_tokenized_unk=221.parquet')
    testset = SequenceDataset(df_test, test=True)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=16, drop_last=False)
    print('# of test smiles:', len(df_test))
    
    model.eval()
    preds = []
    with torch.no_grad():
        for batch_data in tqdm(test_loader):
            x, mask = batch_data
            x = x.to(device)
            mask = mask.to(device)
            with torch.cuda.amp.autocast():
                z = model(x, mask)
                z = torch.sigmoid(z).cpu().numpy()
                preds.append(z)
    
    preds = np.concatenate(preds, 0)
    
    print(preds.shape)    
    submit['binds'] = 0
    submit.loc[submit['protein_name']=='BRD4', 'binds'] = preds[(submit['protein_name']=='BRD4').values, 0]
    submit.loc[submit['protein_name']=='HSA', 'binds'] = preds[(submit['protein_name']=='HSA').values, 1]
    submit.loc[submit['protein_name']=='sEH', 'binds'] = preds[(submit['protein_name']=='sEH').values, 2]
    submit[['id', 'binds']].to_csv(os.path.join(log_dir, 'submission.csv'), index = False)
    
    