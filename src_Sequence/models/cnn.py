
import torch
import torch.nn as nn 

def masked_avg_pooling(tensor, mask):
    mask = mask.unsqueeze(1).expand_as(tensor)  # B * 1 * L -> B * C * L
    tensor = tensor * mask
    
    # 각 시퀀스의 실제 길이만큼 나누어 평균을 구함
    sum_tensor = tensor.sum(dim=2)  # B * C
    mask_sum = mask.sum(dim=2)  # B * C
    avg_tensor = sum_tensor / mask_sum
    
    return avg_tensor


class CNN(nn.Module):
    """
    1D Convolutional Neural Networks that encodes SMILES string.
    """
    def __init__(self, num_embeddings, emb_dim=128, hidden_dim=128 , num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=emb_dim, padding_idx=0)
        self.layer1 = Block(in_channels=emb_dim, out_channels=hidden_dim)
        self.layer2 = Block(in_channels=hidden_dim, out_channels=hidden_dim)
        
        self.mlp = MLP(hidden_dim, num_classes)
        
    def forward(self, x, mask):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)  # Conv1d를 위해 차원 순서 변경
        x = self.layer1(x)
        x = self.layer2(x)
        x = masked_avg_pooling(x, mask)
        x = torch.flatten(x, 1)
        x = self.mlp(x)
    
        return x

    def set_unk_to_zero(self):
        emb_dim = self.embedding.embedding_dim
        # UNK token에 대한 embedding vector를 zero로 초기화
        self.embedding.weight.data[221] = torch.tensor([0.]*emb_dim).float()
    

class Block(nn.Module):
    """
    Bottleneck residual block
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels//2, kernel_size=3, padding=1) 
        self.bn2 = nn.BatchNorm1d(out_channels//2)
        self.conv2 = nn.Conv1d(out_channels//2, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(self.bn1(x)))
        x = self.relu(self.conv2(self.bn2(x))) + residual
        
        return x

class MLP(nn.Module):
    """
    Simple multi layer perceptron layer
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(
                    nn.Linear(in_channels, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.mlp(x)
        
        return x