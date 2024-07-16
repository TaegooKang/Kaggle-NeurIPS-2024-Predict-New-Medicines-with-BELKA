import torch
import numpy as np 
import os
import pandas as pd 

import argparse
from tqdm import tqdm 

from torch.utils.data import DataLoader
from dataset import my_collate_fn_3d, Graph3dTestDataset
from models.leftnet import LEFTNet


import warnings
warnings.simplefilter("ignore")
    
def get_args_parser():
    parser = argparse.ArgumentParser('[Kaggle] Leash-BELKA Competition Training', add_help=False)
    parser.add_argument('--pretrained', type=str)
    
    return parser.parse_args()


def main(args):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LEFTNet(cutoff=10.0, num_layers=4, hidden_channels=64, num_radial=32)
    ckpt = torch.load(args.pretrained, map_location='cpu')['model']
    model.load_state_dict(ckpt)
    model.to(device)
    
    # Inference
    print('Start Inference!!')
    print('Build test graphs...')
    submit = pd.read_parquet('/data/datasets/leash-BELKA/origin/test.parquet')
    
    testset = Graph3dTestDataset()
    test_loader = DataLoader(testset, batch_size=500, shuffle=False, collate_fn=my_collate_fn_3d, num_workers=10)
    
    print('# of test graphs:', len(testset))

    model.eval()
    preds = []
    with torch.no_grad():  
        for batch_data in tqdm(test_loader):
            z, pos, batch = batch_data
            z = z.to(device)
            pos = pos.to(device)
            batch = batch.to(device)
            with torch.cuda.amp.autocast():
                pred = model(z, pos, batch)
                pred = torch.sigmoid(pred).cpu().numpy()
                preds.append(pred)
    
    preds = np.concatenate(preds, 0)
    
    log_dir = os.path.dirname(args.pretrained)
    print(preds.shape)    
    submit['binds'] = 0
    submit.loc[submit['protein_name']=='BRD4', 'binds'] = preds[(submit['protein_name']=='BRD4').values, 0]
    submit.loc[submit['protein_name']=='HSA', 'binds'] = preds[(submit['protein_name']=='HSA').values, 1]
    submit.loc[submit['protein_name']=='sEH', 'binds'] = preds[(submit['protein_name']=='sEH').values, 2]
    submit[['id', 'binds']].to_csv(os.path.join(log_dir, 'submission.csv'), index = False)
        

if __name__ == '__main__':
    
    args = get_args_parser()
    main(args)

