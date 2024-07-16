import torch
import torch.nn as nn

import torch.backends.cudnn as cudnn

import numpy as np 
import os
import math
import sys 

import argparse
import time

import ddp_utils as ddp
import utils
from torch.utils.data import DataLoader
from dataset import build_3dgraph_dataset, my_collate_fn_3d

from models.torchmd import build_torchmdnet
from models.leftnet import LEFTNet
from models.comenet import ComENet

import warnings
warnings.simplefilter("ignore")
    
def get_args_parser():
    parser = argparse.ArgumentParser('[Kaggle] Leash-BELKA Competition Training', add_help=False)
    parser.add_argument('--exp-name', '-ex', default='exp', type=str)
    
    # seed
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--num-workers', default=32, type=int)
    
    # training
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    
    # learning rate 
    parser.add_argument('--base-lr', default=1e-4, type=float)
    
    # optimizer
    parser.add_argument('--weight-decay', default=5e-2, type=float)
    parser.add_argument('--lr-decay-factor', default=0.1, type=float)
    # logging
    parser.add_argument('--log-dir', default='./checkpoints', type=str)
    
    # ddp config
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')
    
    # model
    parser.add_argument('--model', choices=['LEFTNet', 'ComENet', 'TorchMDNet'])
    
    return parser.parse_args()


def main(args):
        
    ddp.init_process_group()
    
    seed = args.seed + ddp.get_rank()
    utils.fix_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    
    trainset, validset = build_3dgraph_dataset()
    
    num_tasks = ddp.get_world_size()
    global_rank = ddp.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        trainset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    sampler_val = torch.utils.data.SequentialSampler(validset)
    
    # In main process
    if ddp.get_rank() == 0:
        # Create logging directory
        log_dir = os.path.join(args.log_dir, args.exp_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        utils.save_args(args, log_dir)
        logger = utils.get_logger(name='train',
                                file_path=os.path.join(log_dir, 'train.log'),
                                stream=True,
                                level='info')
    else:
        logger = None
    
    train_loader = DataLoader(
        trainset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=my_collate_fn_3d
    )

    valid_loader = DataLoader(
        validset, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=my_collate_fn_3d
    )
    
    # model
    if args.model == 'LEFTNet':
        model = LEFTNet(cutoff=10.0, num_layers=4, hidden_channels=64, num_radial=32)
    elif args.model == 'ComENet':
        model = ComENet(num_layers=4, hidden_channels=128, middle_channels=64, out_channels=3)
    elif args.model == 'TorchMDNet':
        model = build_torchmdnet()
    
    model.to(device)
    model_without_ddp = model
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(n_parameters)
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ddp.get_rank()])
    model_without_ddp = model.module
    
    eff_batch_size = args.batch_size * ddp.get_world_size()   
    lr = args.base_lr * eff_batch_size / 256
    
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), 
                                  betas=(0.9, 0.999), lr=lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    criterion = nn.BCEWithLogitsLoss()
    
    decay_step = [12, 24]
    
    # Training procedure
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
    
        for i, batch_data in enumerate(train_loader):
            z, pos, batch, y = batch_data
            z = z.to(device)
            pos = pos.to(device)
            batch = batch.to(device)
            y = y.to(device)
            # forward 
            with torch.cuda.amp.autocast():
                y_hat = model(z, pos, batch)
                loss = criterion(y_hat, y)
            
            # backward
            optimizer.zero_grad()
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward() 
                scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                _ = torch.nn.utils.clip_grad_norm_(model_without_ddp.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                torch.clear_autocast_cache()

            torch.cuda.synchronize()
            train_loss += loss.detach().item()
            
            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)
            
            # logging
            if i % 2000 == 0:
                train_loss_reduced = ddp.all_reduce_mean(train_loss/(i+1))
                if logger is not None:
                    cur = time.time()
                    logger.info(f'[Epoch {epoch+1}/{args.epochs} - {i}iters] Train Loss: {train_loss_reduced:.5f} | Elapsed: {utils.format_time(cur-start)}')
            
            if i == 10000:
                break
            
        #train_loss = ddp.all_reduce_mean(train_loss/len(train_loader))
        train_loss = ddp.all_reduce_mean(train_loss/10000)
        if logger is not None:
            logger.info(f'[Epoch {epoch+1}/{args.epochs}] Train Loss: {train_loss:.5f} | Elapsed: {utils.format_time(cur-start)}')
        
        # multistep lr schedule
        if epoch + 1 in decay_step:
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                param_group['lr'] = lr * args.lr_decay_factor

        if epoch % 4 == 0:
            ## validation
            model.eval()
            val_loss = 0
            preds = []
            ys = []
            stat =  '-----------------------------------------------------\n'
            stat += '|       |   BRD4   |   HSA   |   sEH   |    loss    |\n'
            stat += '-----------------------------------------------------'
            
            with torch.no_grad():
                for i, batch_data in enumerate(valid_loader):
                    z, pos, batch, y = batch_data
                    z = z.to(device)
                    pos = pos.to(device)
                    batch = batch.to(device)
                    y = y.to(device)
        
                    # forward 
                    with torch.cuda.amp.autocast():
                        y_hat = model(z, pos, batch)
                        loss = criterion(y_hat, y)
                        pred = torch.sigmoid(y_hat).cpu().numpy()
                    
                    val_loss += loss.item()
                    
                    preds.append(pred)
                    ys.append(y.cpu().numpy())
                    
                    if i % 2000 == 0:
                        valid_loss_reduced = val_loss/(i+1)
                        if logger is not None:
                            cur = time.time()
                            logger.info(f'[Epoch {epoch+1}/{args.epochs} - {i}iters] Valid Loss: {valid_loss_reduced:.5f} | Elapsed: {utils.format_time(cur-start)}')
                    
            preds = np.concatenate(preds, axis=0)
            ys = np.concatenate(ys, axis=0)
            
            ap_BRD4 = utils.ap(preds[:,0], ys[:,0])
            ap_HSA = utils.ap(preds[:,1], ys[:,1])
            ap_sEH = utils.ap(preds[:,2], ys[:,2])
            
            val_loss /= len(valid_loader)
            stat += f'\n| share |   {ap_BRD4:.3f}  |  {ap_HSA:.3f}  |  {ap_sEH:.3f}  |  {val_loss:.6f}  |'
            stat += '\n-----------------------------------------------------'
            if logger is not None:
                logger.info(f'[Epoch {epoch+1}/{args.epochs}] Validation Stats.. \n' + stat)
        
        if ddp.is_main_process():
            torch.save({'model': model_without_ddp.state_dict()}, os.path.join(log_dir, f'model_{epoch+1}.pth'))
        

if __name__ == '__main__':
    
    args = get_args_parser()
    main(args)

