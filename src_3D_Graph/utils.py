import torch 

import numpy as np
import random
from sklearn.metrics import average_precision_score

import os
import yaml
import logging

# utils ==================================================== #
def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'd '
        i += 1
    if hours > 0 and i <= 3:
        f += str(hours) + 'h '
        i += 1
    if minutes > 0 and i <= 3:
        f += str(minutes) + 'm '
        i += 1
    if secondsf > 0 and i <= 3:
        f += str(secondsf) + 's '
        i += 1
    if millis > 0 and i <= 3:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f                  

def fix_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed) 
    
def ap(preds, targets):
	return average_precision_score(targets, preds, average='micro')
# =========================================================== #

# save arguments
def save_args(args, save_path):
    with open(os.path.join(save_path,'hparams.yaml'), "w", encoding="utf8") as outfile:
        yaml.dump(args, outfile, default_flow_style=False, allow_unicode=True)


def get_logger(name: str, file_path: str, stream=False, level='info')-> logging.RootLogger:

    level_map = {
        'info': logging.INFO,
        'debug': logging.DEBUG
    }
    
    logger = logging.getLogger(name)
    logger.setLevel(level_map[level])  # logging all levels
    
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(file_path)

    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    if stream:
        logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger