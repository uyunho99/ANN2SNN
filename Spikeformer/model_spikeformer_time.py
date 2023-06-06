import torch as th
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
# from spikingjelly.activation_based.neuron import LIFNode

import math

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=192, img_size=32, patch_size=4, drop_rate=0.1):
        super().__init__()
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
        T, B, C, H, W = x.shape # T: time_steps, B: batch_size, C: in_channels, H: img_size, W: img_size
        x = x.flatten(0, 1) # (T*B, C, H, W): (4*32, 3, 32, 32)
        x = self.project(x) # (T*B, embed_dim, num_patches**0.5, num_patches**0.5): (128, 192, 8, 8)
        x = x.flatten(2) # (T*B, embed_dim, num_patches): (128, 192, 64)
        x = x.transpose(1, 2) # (T*B, num_patches, embed_dim): (128, 64, 192)
        x = x.reshape(T, B, -1, self.embed_dim) # (T, B, num_patches, embed_dim): (4, 32, 64, 192)
        
        class_tokens = self.class_token.expand(T, B, -1, -1) # (T, B, 1, embed_dim): (4, 32, 1, 192)
        x = th.cat((class_tokens, x), dim=2) # (T, B, num_patches + 1, embed_dim): (4, 32, 65, 192)
        
        x += self.positional_embedding # (T, B, num_patches + 1, embed_dim): (4, 32, 65, 192)
        # x = self.dropout(x)
        return x

class SpikeSelfAttention(nn.Module):
    def __init__(self, dim=192, num_head=8, qkv_bias=False, dropout=0.1, sr_ratio=1):
        super().__init__()
        self.dim = dim # 임베딩 차원: 192
        self.num_head = num_head # 헤드 개수: 8
        self.scale = (dim // num_head) ** -0.5 # 192 / 8 = 24 -> 1/24
        self.dropout = nn.Dropout(dropout) # 드롭아웃 비율: 0.1

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # (192, 576)
        # self.qkv_batchnorm = nn.BatchNorm1d(dim * 3) # (576)
        self.qkv_neuron = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        # self.qkv_neuron = LIFNode()
        
        self.proj = nn.Linear(dim, dim) # (192, 192)

    def forward(self, x):
        T, B, N, C = x.shape # (T, B, num_patches + 1, embed_dim): (4, 32, 65, 192)
        
        qkv = x.flatten(0, 1) # (T*B, N, C): (128, 65, 192)
        qkv = self.qkv(x).contiguous() # (T*B, N, C*3): (128, 65, 576)
        qkv = qkv.reshape(T, B, N, -1) # (T, B, N, C*3): (4, 32, 65, 576)
        qkv = self.qkv_neuron(qkv).contiguous() # (T, B, N, C*3): (4, 32, 65, 576)
        qkv = qkv.reshape(T, B, N, 3, self.num_head, self.dim // self.num_head) # (T, B, N, 3, num_head, num_head): (4, 32, 65, 3, 8, 24)
        q, k, v = qkv.unbind(dim=3) # (T, B, N, num_head, num_head): (4, 32, 65, 8, 24)

        attn = (q @ k.transpose(-2, -1)) * self.scale # (T, B, N, num_head, num_head): (4, 32, 65, 8, 8)
        attn = attn.softmax(dim=-1) # (T, B, N, num_head, num_head): (4, 32, 65, 8, 8)
        attn = self.dropout(attn) # (T, B, N, num_head, num_head): (4, 32, 65, 8, 8)
        
        x = (attn @ v).transpose(2, 3).reshape(T, B, N, C) # T, B, N, C: (4, 32, 65, 192)
        x = self.proj(x) # (T, B, N, C): (4, 32, 65, 192)
        x = self.dropout(x) # (T, B, N, C): (4, 32, 65, 192)
        return x 

class MultiLayerPerceptron(nn.Module):
    def __init__(self, in_features, hidden_features, output_features, bias=True, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias) # (192, 768)
        self.fc2 = nn.Linear(hidden_features, output_features, bias=bias) # (768, 192)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x) # (4, 32, 65, 192) -> (4, 32, 65, 768) 
        x = self.activation(x) 
        x = self.dropout(x)
        x = self.fc2(x) # (4, 32, 65, 768) -> (4, 32, 65, 192)
        x = self.activation(x)
        x = self.dropout(x)
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, dropout=0.1, drop_path=0., sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attention = SpikeSelfAttention(dim, num_heads, qkv_bias, dropout) # (192, 8)
        self.mlp = MultiLayerPerceptron(dim, int(dim * mlp_ratio), dim, dropout=dropout) # (192, 768, 192)
        
    def forward(self, x):
        x = x + self.attention(x) 
        x = x + self.mlp(x) 
        return x
        
class SpikingViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192, num_heads=8, 
                 mlp_ratio=4., qkv_bias=False, dropout=0.1, num_classes=10, 
                 time_steps=4, drop_path_rate=0.1, depths=10, sr_ratios=[8, 4, 2]):
        super().__init__()
        self.num_classes = num_classes # 10
        self.patch_size = patch_size # 4
        self.embed_dim = embed_dim # 192
        self.num_patches = (img_size // patch_size) ** 2 # 8*8 = 64
        
        self.T = time_steps # 4
        dpr = [x.item() for x in th.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embedding = PatchEmbedding(in_channels, embed_dim, img_size, patch_size, dropout) # (3, 192, 32, 4, 0.1)
        blocks = nn.Sequential(*[
            TransformerEncoder(embed_dim, num_heads, mlp_ratio, qkv_bias, dropout, 
                               drop_path=dpr[i], sr_ratio=sr_ratios) for i in range(depths) # (192, 8, 4.0, False, 0.1) * 12
        ])
        
        setattr(self, f"patch_embedding", patch_embedding)
        setattr(self, f"blocks", blocks)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) # (192, 10)
        self.apply(self._init_weights)

    @th.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)
            
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        blocks = getattr(self, f"blocks")
        patch_embedding = getattr(self, f"patch_embedding")

        x = patch_embedding(x)
        for block in blocks:
            x = block(x)
        return x.mean(2)

    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x