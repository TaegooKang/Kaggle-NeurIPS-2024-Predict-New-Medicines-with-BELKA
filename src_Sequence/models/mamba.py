import torch
import torch.nn as nn 

from mamba_ssm import Mamba

def masked_avg_pooling(x, mask):
    # x: (batch_size, sequence_length, feature_dim)
    # mask: (batch_size, sequence_length)
    
    # 마스크를 x의 크기에 맞게 확장
    mask = mask.unsqueeze(-1).expand_as(x)  # (batch_size, sequence_length, feature_dim)
    
    # 마스크를 이용하여 패딩 부분을 0으로 설정
    x = x * mask  # (batch_size, sequence_length, feature_dim)
    
    # 마스크의 합을 구하여 시퀀스의 실제 길이를 계산
    mask_sum = mask.sum(dim=1, keepdim=True)  # (batch_size, 1, feature_dim)
    
    # 마스크 합이 0인 부분을 방지하기 위해 1로 설정 (분모가 0이 되는 것을 방지)
    mask_sum = torch.clamp(mask_sum, min=1e-9)
    
    # 패딩을 제외한 부분의 합을 계산하고 평균을 구함
    pooled = x.sum(dim=1) / mask_sum.squeeze(1)  # (batch_size, feature_dim)
    
    return pooled


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
    

class MambaLM(nn.Module):
    
    def __init__(self, num_embeddings, emb_dim=128, hidden_dim=128, num_layers=4, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=emb_dim, padding_idx=0)
        self.conv_stem = nn.Sequential(
            nn.BatchNorm1d(emb_dim),
            nn.Conv1d(emb_dim, hidden_dim, 3, 1, 1),
            nn.ReLU()
        )
        self.mamba_layers = nn.ModuleList(
            [Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList(
            [RMSNorm(hidden_dim) for _ in range(num_layers)]
        )
        
        # self.lin = nn.Linear(hidden_dim, num_classes, bias=False)
        
        self.lin = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
        
    def forward(self, x, mask):
        x = self.embedding(x)
        x = x.permute(0,2,1)  # Conv1d를 위해 차원 순서 변경
        x = self.conv_stem(x)
        x = x.permute(0,2,1)
    
        for norm, layer in zip(self.norms, self.mamba_layers):
            x = layer(norm(x)) + x
        
        x = masked_avg_pooling(x, mask)
        
        x = self.lin(x)
    
        return x
    
    
    
if __name__ == '__main__':
    
    x = torch.tensor([[2, 11, 5,6,7,5, 0, 0],
                      [1, 3, 4,3,2,0, 0, 0]], dtype=torch.long).cuda()

    # 마스크 (0은 패딩된 부분, 1은 실제 데이터 부분)
    mask = torch.tensor([[1, 1, 1, 1,1,1, 0, 0],
                        [1, 1, 1,1,1,0, 0, 0]], dtype=torch.float).cuda()
    
    model = MambaLM(15, 64, 64, 3).cuda()
    
    z = model(x, mask)
    print(z)