import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

import math

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=192, img_size=32, patch_size=4, drop_rate=0.1):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size # 32
        self.patch_size = patch_size # 4
        self.num_patches = (img_size // patch_size) ** 2 # 8*8 = 64
        self.embed_dim = embed_dim # 192
        # self.dropout = nn.Dropout(drop_rate) # 0.1
        
        self.project = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        
        self.class_token = nn.Parameter(th.zeros(1, 1, embed_dim)) # (1, 1, 192)
        self.num_patches += 1 # 65
        self.positional_embedding = nn.Parameter(th.zeros(1, self.num_patches, embed_dim)) # (1, 65, 192)
        
        # nn.init.normal_(self.class_token, std=0.02) # class_token 초기화
        # trunc_normal_(self.positional_embedding, std=.02) # positional_embedding 초기화

    def forward(self, x):
        B, C, H, W = x.shape # (batch_size, in_channels, img_size, img_size): (batch_size, 3, 32, 32)
        x = self.project(x)  # (batch_size, embed_dim, num_patches**0.5, num_patches**0.5): (batch_size, 192, 8, 8)
        x = x.flatten(2)  # (batch_size, embed_dim, num_patches): (batch_size, 192, 64)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim): (batch_size, 64, 192)
        
        class_tokens = self.class_token.expand(B, -1, -1)  # (batch_size, 1, embed_dim): (batch_size, 1, 192)
        x = th.cat((class_tokens, x), dim=1)  # (batch_size, num_patches + 1, embed_dim): (batch_size, 65, 192)
        
        x += self.positional_embedding  # (batch_size, num_patches + 1, embed_dim): (batch_size, 65, 192)
        # x = self.dropout(x)
        return x

class SpikeSelfAttention(nn.Module):
    def __init__(self, dim=192, num_head=8, qkv_bias=False, dropout=0.1):
        super().__init__()
        self.dim = dim # 임베딩 차원: 192
        self.num_head = num_head # 헤드 개수: 8
        self.scale = (dim // num_head) ** -0.5 # 192 / 8 = 24 -> 1/24
        self.dropout = dropout # 드롭아웃 비율: 0.1

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # (192, 576)
        self.qkv_batchnorm = nn.BatchNorm1d(dim * 3)
        self.qkv_neuron = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        
        self.proj = nn.Linear(dim, dim) # (192, 192)
        
        # # 쿼리
        # self.query = nn.Linear(dim, dim)
        # self.query_batchnorm = nn.BatchNorm1d(dim)
        # self.query_neuron = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        
        # # 키
        # self.key = nn.Linear(dim, dim)
        # self.key_batchnorm = nn.BatchNorm1d(dim)
        # self.key_neuron = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        
        # # 값
        # self.value = nn.Linear(dim, dim)
        # self.value_batchnorm = nn.BatchNorm1d(dim)
        # self.value_neuron = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        # # 출력
        # self.out = nn.Linear(dim, dim)
        # self.out_batchnorm = nn.BatchNorm1d(dim)
        # self.out_neuron = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x):
        B, N, C = x.shape # (batch_size, num_patches + 1, embed_dim): (batch_size, 65, 192)

        qkv = self.qkv_neuron(self.qkv_batchnorm(self.qkv(x))) # (batch_size, num_patches + 1, embed_dim * 3): (batch_size, 65, 576)
        qkv = qkv.reshape(B, N, 3, self.num_head, C // self.num_head).permute(2, 0, 3, 1, 4) # (3, batch_size, num_head, num_patches + 1, embed_dim // num_head): (3, batch_size, 8, 65, 24)
        q, k, v = qkv.unbind(dim=0) # (batch_size, num_head, num_patches + 1, embed_dim // num_head): (batch_size, 8, 65, 24)
        # q = self.neuron(self.query(x)) 
        # k = self.neuron(self.key(x))
        # v = self.neuron(self.value(x))

        attn = (q @ k.transpose(-2, -1)) * self.scale # (batch_size, num_head, num_patches + 1, num_patches + 1): (batch_size, 8, 65, 65)
        attn = attn.softmax(dim=-1) # (batch_size, num_head, num_patches + 1, num_patches + 1): (batch_size, 8, 65, 65)
        attn = self.dropout(attn) # (batch_size, num_head, num_patches + 1, num_patches + 1): (batch_size, 8, 65, 65)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # (batch_size, num_patches + 1, embed_dim): (batch_size, 65, 192)
        x = self.proj(x) # (batch_size, num_patches + 1, embed_dim): (batch_size, 65, 192)
        x = self.dropout(x) # (batch_size, num_patches + 1, embed_dim): (batch_size, 65, 192)
        return x # (batch_size, num_patches + 1, embed_dim): (batch_size, 65, 192)

class MultiLayerPerceptron(nn.Module):
    def __init__(self, in_features, hidden_features, output_features, bias=True, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias) # (192, 768)
        self.fc2 = nn.Linear(hidden_features, output_features, bias=bias) # (768, 192)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x) # (batch_size, num_patches + 1, hidden_features): (batch_size, 65, 768)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x) # (batch_size, num_patches + 1, output_features): (batch_size, 65, 192)
        x = self.activation(x)
        x = self.dropout(x)
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, dropout=0.1,
                 activation=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attention = SpikeSelfAttention(dim, num_heads, qkv_bias, dropout) # (192, 8)
        self.mlp = MultiLayerPerceptron(dim, int(dim * mlp_ratio), dim, dropout=dropout) # (192, 768, 192)
        
    def forward(self, x):
        x += self.attention(self.norm1(x))
        x += self.mlp(self.norm2(x))
        return x
        
class SpikingViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192, depth=12, num_heads=8, 
                 mlp_ratio=4., qkv_bias=False, dropout=0.1, num_classes=10):
        super().__init__()
        self.num_classes = num_classes # 10
        self.patch_size = patch_size # 4
        self.embed_dim = embed_dim # 192
        self.num_patches = (img_size // patch_size) ** 2 # 8*8 = 64

        self.patch_embedding = PatchEmbedding(in_channels, embed_dim, img_size, patch_size, dropout) # (3, 192, 32, 4, 0.1)
        self.blocks = nn.Sequential(*[
            TransformerEncoder(embed_dim, num_heads, mlp_ratio, qkv_bias, dropout) for _ in range(depth) # (192, 8, 4.0, False, 0.1) * 12
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) # (192, 10)
        
    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.head(x)[:, 0]
        return x