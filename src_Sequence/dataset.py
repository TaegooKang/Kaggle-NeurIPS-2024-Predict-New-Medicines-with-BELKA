import torch
from torch.utils.data import Dataset


MAX_LEN = 150

class SequenceDataset(Dataset):
    def __init__(self, df, test=False): 
        self.df = df
        self.test = test
        
    def __getitem__(self, index):
        v = self.df.iloc[index].values
        
        if not self.test:
            x, y = v[:MAX_LEN], v[MAX_LEN:]
            mask = x != 0
            x = torch.tensor(x, dtype=torch.long) 
            y = torch.tensor(y, dtype=torch.float)
            mask = torch.tensor(mask, dtype=torch.float)
            
            return x, mask, y
        
        else:
            mask = v != 0
            x = torch.tensor(v, dtype=torch.long) 
            mask = torch.tensor(mask, dtype=torch.float)
            
            return x, mask
    
    def __len__(self):
        return len(self.df)        
    

    
    