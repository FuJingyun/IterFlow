import torch.nn as nn
import torch
import sys
import numpy as np
import os
import torch.nn.functional as F
# from utils.model_utils import *
# from utils import *

sys.path.append("/home/fjy/DeFlow-master/scripts/")
from caonet_utils.model_utils import *
from caonet_utils import *
# sys.path.append('/home/fjy/DeFlow-master/scripts/cmflow_utils/')
# from lib import pointnet2_utils as pointutils
from scripts.raflow_utils.lib import pointnet2_utils as pointutils

# from timm.models.layers import DropPath,trunc_normal_
import math
# from mamba_ssm import Mamba
LEAKY_RATE = 0.1
use_bn = False

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_activation=True,
                 use_leaky=True, bn=use_bn):
        super(Conv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if use_activation:
            relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        else:
            relu = nn.Identity()

        self.composed_module = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm1d(out_channels) if bn else nn.Identity(),
            relu
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.composed_module(x)
        x = x.permute(0, 2, 1)
        return x


class SubFold(nn.Module):
    def __init__(self, in_channel , step , hidden_dim = 512):
        super().__init__()
        self.in_channel = in_channel
        self.step = step
        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )
        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

    def forward(self, x, c):
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
        seed = c.to(x.device) # b 3 n2
        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)
        return fd2




class old_GeoCrossAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=1, qkv_bias=False, qk_scale=1, attn_drop=0., proj_drop=0.,
                 aggregate_dim=16):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        # self.q_map = nn.Identity()  # nn.Linear(dim, out_dim, bias=qkv_bias)
        # self.k_map = nn.Identity()  # nn.Linear(dim, out_dim, bias=qkv_bias)
        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)  # nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)  # nn.Linear(dim, out_dim, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)

        self.x_map = nn.Identity()  # nn.Linear(aggregate_dim, 1)
        # self.softmax = nn.Softmax(dim=3)

    def forward(self, q, k, v):
        B, N, _ = q.shape
        C = self.out_dim
        NK = k.size(1)
        
        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.view(B, NK, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # attn = self.softmax(attn)
        attn = attn.detach()
        x = (attn @ v).transpose(1, 2).reshape(B, N, 3)

        x = self.x_map(x)

        return x
    


class GeoCrossAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=1, qkv_bias=False, qk_scale=1, attn_drop=0., proj_drop=0.,
                 aggregate_dim=16):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        # self.scale = qk_scale or head_dim ** -0.5

        # self.q_map = nn.Identity()  # nn.Linear(dim, out_dim, bias=qkv_bias)
        # self.k_map = nn.Identity()  # nn.Linear(dim, out_dim, bias=qkv_bias)
        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)  # nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)  # nn.Linear(dim, out_dim, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)

        # self.x_map = nn.Identity()  # nn.Linear(aggregate_dim, 1)
        self.softmax = nn.Softmax(dim=3)

    def forward(self, q, k, v):
        B, N, _ = q.shape
        C = self.out_dim
        NK = k.size(1)
        # q: B N C
        # K: B N C
        # V: B 3 N 
        
        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.view(B, NK, self.num_heads, -1).permute(0, 2, 1, 3)

        # q: B 1 N C
        # K: B 1 N C
        # V: B 1 N 3

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = (q @ k.transpose(-2, -1))
        # torch.Size([8, 1, 64, 64])
        # attn = attn.softmax(dim=-1)
        

        # attn: B 1 N N
        attn = self.softmax(attn)
        # attn = attn.detach()
        # B 1 N 3  ->   B N 1 3  ->  B N 3
        x = (attn @ v).transpose(1, 2).reshape(B, N, 3)
        #  x = self.x_map(x)
        return x





class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_pred=64, num_point=128):
        super().__init__()
        # self.mlp_convs = nn.ModuleList()
        # self.mlp_bns = nn.ModuleList()
        # self.mlp2_convs = nn.ModuleList()
        # self.mlp2_bns = nn.ModuleList()
        # self.mlp3_convs = nn.ModuleList()
        # self.mlp3_bns = nn.ModuleList()
        self.dim = dim

        self.queryandgroup = pointutils.QueryAndGroup(radius=4.0, nsample=8)
        # self.norm_q = nn.Identity()  # norm_layer(dim)
        # self.norm_k = nn.Identity()  # norm_layer(dim)
        self.norm_q = norm_layer(dim)  # norm_layer(dim)
        self.norm_k = norm_layer(dim)  # norm_layer(dim)
        self.attn = GeoCrossAttention(dim, dim, num_heads=1, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                                      proj_drop=drop, aggregate_dim=16)

        # self.fold_step = int(pow(num_pred, 0.5) + 0.5)
        # self.generate_anchor = SubFold(dim, step=self.fold_step, hidden_dim=dim // 2)

    def forward(self, input_feature, x, coor):
        # B, _, _ = x.size()
        idx = pointutils.furthest_point_sample(coor.contiguous(), 64).unsqueeze(-1).to(torch.int64)
        sample_coor = torch.gather(coor, 1, idx.repeat(1, 1, 3))  # [B , 64, 3]
        sample_x = torch.gather(x, 1, idx.repeat(1, 1, self.dim))  # [B, 64, 256]
        sample_input_feature = torch.gather(input_feature, 1, idx.repeat(1, 1, 3))

        local_x = self.queryandgroup(coor.contiguous(), sample_coor.contiguous(), x.permute(0, 2, 1).contiguous())
        local_coor = local_x[:, :3, :, :]   # [B, xyz, 64, nsample]  nsample=8
        local_x = local_x[:, 3:, :, :]  # [B, 256, 64, nsample]  nsample=8

        local_input_feature = self.queryandgroup(coor.contiguous(), sample_coor.contiguous(), input_feature.permute(0, 2, 1).contiguous())
        local_input_feature = local_input_feature[:, 3:, :, :]
        diff_input_feature =  torch.mean(local_input_feature, -1)

        diff_coor = torch.mean(local_coor, -1)  # [B, xyz, 64]

        global_x = torch.max(local_x, -1)[0]  # [B, 256, 64]
        # diff_x = (global_x - sample_x.permute(0, 2, 1)).unsqueeze(2)  # [B, 256, 64] -> [B, 256, 1, 64]
        # x_2 = diff_x.squeeze(2).permute(0, 2, 1)
        diff_x = global_x - sample_x.permute(0, 2, 1)  # [B, 256, 64] 
        x_2 = diff_x.permute(0, 2, 1)  # [B, 64, 256] 
                
        norm_k = self.norm_k(sample_x)  # B N dim   [B, 64, 256]
        norm_q = self.norm_q(x_2)  # B L dim  [B, 64, 256]

        # coor_2 [B , 64, 3]
        coor_2 = self.attn(q=norm_q, k=norm_k, v=diff_coor)   # diff_coor [B, xyz, 64]
        input_feature_2 = self.attn(q=norm_q, k=norm_k, v=diff_input_feature)

        sample_coor = sample_coor + coor_2  # [B , 64, 3]
        sample_x = sample_x + x_2 # [B, 64, 256]
        sample_input_feature = sample_input_feature + input_feature_2
        return sample_x, sample_coor, sample_input_feature







class simple_EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_pred=64, num_point=128):
        super().__init__()
        # self.mlp_convs = nn.ModuleList()
        # self.mlp_bns = nn.ModuleList()
        # self.mlp2_convs = nn.ModuleList()
        # self.mlp2_bns = nn.ModuleList()
        # self.mlp3_convs = nn.ModuleList()
        # self.mlp3_bns = nn.ModuleList()
        self.dim = dim

        self.queryandgroup = pointutils.QueryAndGroup(radius=4.0, nsample=8)
        # self.norm_q = nn.Identity()  # norm_layer(dim)
        # self.norm_k = nn.Identity()  # norm_layer(dim)
        self.norm_q = norm_layer(dim)  # norm_layer(dim)
        self.norm_k = norm_layer(dim)  # norm_layer(dim)
        self.attn = GeoCrossAttention(dim, dim, num_heads=1, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                                      proj_drop=drop, aggregate_dim=16)

        # self.head_num = 2
        # self.atten_fusion1 = torch.nn.MultiheadAttention(embed_dim=128, num_heads=self.head_num, dropout=0, batch_first=True)

        # self.fold_step = int(pow(num_pred, 0.5) + 0.5)
        # self.generate_anchor = SubFold(dim, step=self.fold_step, hidden_dim=dim // 2)

    def forward(self, input_feature, x, coor):
        # B, _, _ = x.size()
        idx = pointutils.furthest_point_sample(coor.contiguous(), 64).unsqueeze(-1).to(torch.int64)
        sample_coor = torch.gather(coor, 1, idx.repeat(1, 1, 3))  # [B , 64, 3]
        sample_x = torch.gather(x, 1, idx.repeat(1, 1, self.dim))  # [B, 64, 256]
        sample_input_feature = torch.gather(input_feature, 1, idx.repeat(1, 1, 3))

        local_x = self.queryandgroup(coor.contiguous(), sample_coor.contiguous(), x.permute(0, 2, 1).contiguous())
        local_coor = local_x[:, :3, :, :]   # [B, xyz, 64, nsample]  nsample=8
        local_x = local_x[:, 3:, :, :]  # [B, 256, 64, nsample]  nsample=8

        local_input_feature = self.queryandgroup(coor.contiguous(), sample_coor.contiguous(), input_feature.permute(0, 2, 1).contiguous())
        local_input_feature = local_input_feature[:, 3:, :, :]

        local_input_feature = local_input_feature.detach()
        local_coor = local_coor.detach()
        local_x = local_x.detach()
        
        diff_input_feature =  torch.mean(local_input_feature, -1)  # [B, 3, 64]
        diff_coor = torch.mean(local_coor, -1)  # [B, xyz, 64]
        diff_x = torch.mean(local_x, -1)  # [B, 256, 64]

        global_x = torch.max(local_x, -1)[0]  # [B, 256, 64]
        # diff_x = (global_x - sample_x.permute(0, 2, 1)).unsqueeze(2)  # [B, 256, 64] -> [B, 256, 1, 64]
        # x_2 = diff_x.squeeze(2).permute(0, 2, 1)
        diff_x = global_x - sample_x.permute(0, 2, 1)  # [B, 256, 64] 
        x_2 = diff_x.permute(0, 2, 1)  # [B, 64, 256] 
                
        norm_k = self.norm_k(sample_x)  # B N dim   [B, 64, 256]
        norm_q = self.norm_q(x_2)  # B L dim  [B, 64, 256]

        # coor_2 [B , 64, 3]
        coor_2 = self.attn(q=norm_q, k=norm_k, v=diff_coor)   # diff_coor [B, xyz, 64]
        # coor_2 = self.atten_fusion1(norm_q, norm_k, diff_coor.permute(0, 2, 1))
        input_feature_2 = self.attn(q=norm_q, k=norm_k, v=diff_input_feature)

        sample_coor = sample_coor + coor_2  # [B , 64, 3]
        sample_x = sample_x + x_2 # [B, 64, 256]
        sample_input_feature = sample_input_feature + input_feature_2
       
        # return norm_q, coor_2, diff_input_feature.permute(0, 2, 1)
        return sample_x, sample_coor, sample_input_feature





