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

class SSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.q_linear = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.k_linear = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.v_linear = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='torch')

        self.proj_linear = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

    def forward(self, x):
        T,B,N,C = x.shape

        x_for_qkv = x.flatten(0, 1)  # TB, N, C
        q_linear_out = self.q_linear(x_for_qkv)  # [TB, N, C]
        q_linear_out = self.q_bn(q_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        q_linear_out = self.q_lif(q_linear_out)
        q = q_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_linear_out = self.k_linear(x_for_qkv)
        k_linear_out = self.k_bn(k_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        k_linear_out = self.k_lif(k_linear_out)
        k = k_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_linear_out = self.v_linear(x_for_qkv)
        v_linear_out = self.v_bn(v_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        v_linear_out = self.v_lif(v_linear_out)
        v = v_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = attn @ v
        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_lif(self.proj_bn(self.proj_linear(x).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C))
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
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, dropout=0.1, drop_path=0., sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attention = SSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=dropout, proj_drop=dropout, sr_ratio=sr_ratio) # (192, 8, False, 0.1)
        self.mlp = MultiLayerPerceptron(dim, int(dim * mlp_ratio), dim, dropout=dropout) # (192, 768, 192)
        
    def forward(self, x):
        x = x + self.attention(x) 
        x = x + self.mlp(x) 
        return x
        
class SpikingViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192, num_heads=8, 
                 mlp_ratio=4., qkv_bias=False, dropout=0.1, num_classes=10, 
                 time_steps=4, drop_path_rate=0.1, depths=5, sr_ratios=[8, 4, 2]):
        super().__init__()
        self.num_classes = num_classes # 10
        self.patch_size = patch_size # 4
        self.embed_dim = embed_dim # 192
        self.num_patches = (img_size // patch_size) ** 2 # 8*8 = 64
        
        self.T = time_steps # 4
        dpr = [x.item() for x in th.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        self.patch_embedding = PatchEmbedding(in_channels, embed_dim, img_size, patch_size, dropout) # (3, 192, 32, 4, 0.1)
        # self.blocks = nn.Sequential(*[
        #     TransformerEncoder(embed_dim, num_heads, mlp_ratio, qkv_bias, dropout, 
        #                        drop_path=dpr[i], sr_ratio=sr_ratios) for i in range(depths) # (192, 8, 4.0, False, 0.1) * 12
        # ])
        self.block = TransformerEncoder(embed_dim, num_heads, mlp_ratio, qkv_bias, dropout)
        # setattr(self, f"patch_embedding", patch_embedding)
        # setattr(self, f"blocks", blocks)
        
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
        # blocks = getattr(self, f"blocks")
        # patch_embedding = getattr(self, f"patch_embedding")

        x = self.patch_embedding(x)
        # for block in blocks:
        #    x = block(x)
        x = self.block(x)
        return x.mean(2)

    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x