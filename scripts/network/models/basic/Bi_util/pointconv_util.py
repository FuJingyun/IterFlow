
import torch 
import torch.nn as nn 

import torch.nn.functional as F
from time import time
import numpy as np
# from sklearn.neighbors.kde import KernelDensity
from scripts.network.models.basic.Bi_util.pointnet2 import pointnet2_utils


import pointops_cuda

LEAKY_RATE = 0.1
use_bn = False

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=use_bn):
        super(Conv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        self.composed_module = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm1d(out_channels) if bn else nn.Identity(),
            relu
        )

    def forward(self, x):
        x = self.composed_module(x)
        return x

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=use_bn, bias=True):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        self.composed_module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            relu
        )

    def forward(self, x):
        x = self.composed_module(x)
        return x
    
    
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist



def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx




# def get_offset_knn(input):
#     #xyz1 : B, N, 3
#     b, max_points, _ = input.size()
#     if b>1:
#         n_o, count = [ max_points ],max_points
#         for i in range(1, b):
#             count += max_points
#             # n_o 记录每一个点云末尾数量标号， 如 [8192, 16384, 24576, 32768]
#             n_o.append(count)
#         n_o = torch.cuda.IntTensor(n_o)
 
#     else:
#         n_o = [ max_points ]
#         n_o = torch.cuda.IntTensor(n_o)
#     return n_o # (b)


def knn_point_new(nsample, xyz, new_xyz, offset=None, new_offset=None):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]

    needed input: xyz: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)
    original output: idx: (m, nsample), dist2: (m, nsample)
    """
    if new_xyz is None: new_xyz = xyz
    assert xyz.is_contiguous() and new_xyz.is_contiguous()

    B_in, N_in, C_in = xyz.shape 
    _, S_in, _ = new_xyz.shape
    
    if offset is None:
        # offset =  get_offset_knn(xyz)
        offset = np.linspace(N_in,B_in*N_in,B_in, dtype = int)
        # offset = torch.from_numpy(offset).int().cuda()
        offset = torch.cuda.IntTensor(offset)

    if new_offset is None:
        # new_offset =  get_offset_knn(new_xyz)
        new_offset = np.linspace(S_in,B_in*S_in,B_in, dtype = int)
        # new_offset = torch.from_numpy(new_offset).int().cuda()
        new_offset = torch.cuda.IntTensor(new_offset)

    xyz = xyz.view(B_in*N_in,C_in)
    new_xyz = new_xyz.view(B_in*S_in,C_in)

 
    m = new_xyz.shape[0]
    idx = torch.cuda.IntTensor(m, nsample).zero_()
    dist2 = torch.cuda.FloatTensor(m, nsample).zero_()
    pointops_cuda.knnquery_cuda(m, nsample, xyz, new_xyz, offset, new_offset, idx, dist2)

    idx = idx.view(B_in,S_in,nsample)
    # dist2 = torch.sqrt(dist2)
    # dist2_out = dist2.view(B_in,S_in,nsample)

    # return idx, dist2_out
    return idx


def index_points_gather(points, fps_idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """

    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.gather_operation(points_flipped, fps_idx)
    return new_points.permute(0, 2, 1).contiguous()

def index_points_group(points, knn_idx):
    """
    Input:
        points: input points data, [B, N, C]
        knn_idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.grouping_operation(points_flipped, knn_idx.int()).permute(0, 2, 3, 1)

    return new_points

def group(nsample, xyz, points):
    """
    Input:
        nsample: scalar
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        # new_xyz: sampled points position data, [B, N, C]
        new_points: sampled points data, [B, npoint, nsample, C+D]
    """
    B, N, C = xyz.shape
    S = N
    new_xyz = xyz
    idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points_group(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C) # [B, npoint, nsample, C]
    if points is not None:
        grouped_points = index_points_group(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm

def group_query(nsample, s_xyz, xyz, s_points):
    """
    Input:
        nsample: scalar
        s_xyz: input points position data, [B, N, C]
        s_points: input points data, [B, N, D]
        xyz: input points position data, [B, S, C]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]

        new_points: sampled points position data, [B, S, K, C+D]
        grouped_xyz_norm: sampled points data, [B, S, K, C]
    """
    B, N, C = s_xyz.shape
    S = xyz.shape[1]
    new_xyz = xyz
    idx = knn_point(nsample, s_xyz, new_xyz) # [B, npoint, nsample]
    grouped_xyz = index_points_group(s_xyz, idx) # [B, npoint, nsample, C]
    # 结合broadcast机制，获得相对特征 # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if s_points is not None:
        grouped_points = index_points_group(s_points, idx) # [B, npoint, nsample, D]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm # [B, npoint, nsample, C]

    return new_points, grouped_xyz_norm




def new_group_query(nsample, s_xyz, xyz, s_points,use_xyz=True):
    """
    Input:
        nsample: scalar
        s_xyz: input points position data, [B, N, C]
        s_points: input points data, [B, N, D]
        xyz: input points position data, [B, S, C]
    Return:
        new_points: sampled points data, [B, npoint, nsample, D or D+3]
    """
    B, N, C = s_xyz.shape
    S = xyz.shape[1]
    new_xyz = xyz
    idx = knn_point(nsample, s_xyz, new_xyz) # [B, S, nsample]   

    if use_xyz:
        grouped_xyz = index_points_group(s_xyz, idx) # [B, S, nsample, C=3]
        # 结合broadcast机制，获得相对坐标 # [B, S, nsample, C=3]
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
        grouped_points = index_points_group(s_points, idx) # [B, S, nsample, D]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, S, nsample, C+D]
    else:
        new_points = index_points_group(s_points, idx) # [B, S, nsample, D]
    # return new_points, grouped_xyz_norm
    return new_points


class KNN_WeightNet(nn.Module):
    """
        Input:
            knn_feature:  [B, N, K, C]
        Return:
            knn_idx: [B, N, topk]
        """
    def __init__(self, in_planes, topk):
        super(KNN_WeightNet, self).__init__()
        self.mid_plane1 = 32
        self.mid_plane2 = 16
        self.mid_plane3 = 8
        self.mid_plane4 = 1
        self.topk = topk

        self.linear = nn.Sequential(nn.Linear(in_planes, self.mid_plane1), nn.Linear(self.mid_plane1,self.mid_plane2), 
                                    nn.ReLU(inplace=True), nn.Linear(self.mid_plane2,self.mid_plane3), 
                                    nn.ReLU(inplace=True), nn.Linear(self.mid_plane3,self.mid_plane4), nn.ReLU(inplace=True))

        # self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes))
        # self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
        #                             nn.Linear(mid_planes, mid_planes // share_planes),
        #                             nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
        #                             nn.Linear(mid_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, knn_feature):
        # w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes, self.mid_planes).sum(2)  # (n, nsample, C_mid)
        # for i, layer in enumerate(self.linear_w): w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
        # w = self.softmax(w)
        # x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
        # return x
        B, N, K, _ = knn_feature.shape
        for i, layer in enumerate(self.linear): knn_feature = layer(knn_feature)
        knn_feature = knn_feature.view(B,N,K)
        knn_feature = self.softmax(knn_feature)
        final_topk = torch.topk(knn_feature.clone(), k = self.topk, dim=2, sorted=True) 
        return final_topk.indices





class WeightNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit = [8, 8], bn = use_bn):
        super(WeightNet, self).__init__()

        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        
    def forward(self, localized_xyz):
        #xyz : BxCxKxN

        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                weights =  F.relu(bn(conv(weights)))
            else:
                weights = F.relu(conv(weights))

        return weights

class PointConv(nn.Module):
    def __init__(self, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(PointConv, self).__init__()
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def forward(self, xyz, points):
        """
        PointConv without strides size, i.e., the input and output have the same number of points.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1) # B N 3
        points = points.permute(0, 2, 1) # B N 512

        # new_points :[B, npoint, nsample, 3+C]
        # grouped_xyz_norm : [B, npoint, nsample, 3]
        new_points, grouped_xyz_norm = group(self.nsample, xyz, points) # [B, npoint, nsample, C+D]

        # [B, C=3 , nsample=9, npoint=256]
        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        # [B, C=16 , nsample=9, npoint=256]
        weights = self.weightnet(grouped_xyz) #BxWxKxN
        # input : [B, npoint, C+D, nsample]  other: [B 256 9 16]
        # matmul: B 256 (C+D)*16
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other = weights.permute(0, 3, 2, 1)).view(B, N, -1) #BxNxWxK * BxNxKxC => BxNxWxC -> BxNx(W*C)
        # B 256 128
        new_points = self.linear(new_points)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)
        # B 128 256
        new_points = self.relu(new_points)

        return new_points

class PointConvD(nn.Module):
    def __init__(self, npoint, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(PointConvD, self).__init__()
        self.npoint = npoint
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz, points):
        """
        PointConv with downsampling.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        #import ipdb; ipdb.set_trace()
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1).contiguous() # B N C
        points = points.permute(0, 2, 1).contiguous() # B N D 

        # 降采样至 npoint 个点 (2048)
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = index_points_gather(xyz, fps_idx) # [B, S, C]

        """
        new_points: sampled points position data, [B, S, K, C+D] # [B, npoint, nsample, C+D]
        grouped_xyz_norm: sampled points data, [B, S, K, C] # [B, npoint, nsample, C]
        """
        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, points)

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1) # [B,C,K,S]
        weights = self.weightnet(grouped_xyz) # 通过一系列二维卷积（+BN）变成# [B,16,K,S]
        # B, N, S, C
        # input： B S C+D K ；other B S K 16--> B S C+D 16 --> B S 16*(C+D)
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other = weights.permute(0, 3, 2, 1)).view(B, self.npoint, -1)
        # [B, S, D'] D'= 16 *(64+3)    # self.linear : weightnet * in_channel
        new_points = self.linear(new_points)
        # [B, S, D'] 
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1) # [B, D', S]

        new_points = self.relu(new_points)

        return new_xyz.permute(0, 2, 1), new_points, fps_idx
    


class TransitionDown_l0(nn.Module):
    """
        TransitionDown_l0 with stride 1.
        Input:
            color: input points data, [B, N, D ]
        Return:
            new_color: sampled points position data, [B, N, D']
        """
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, color):
        new_color = self.linear(color) # [B, N, D']
        # 当input的维度为（N, C）时，BN将对C维归一化；当input的维度为(N, C, L) 时，归一化的维度同样为C维
        new_color = self.bn(new_color.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        new_color = self.relu(new_color)  # [B, N, D']
        return new_color
    


class new_PointConvD(nn.Module):
    def __init__(self, npoint, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(new_PointConvD, self).__init__()
        self.npoint = npoint
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz, points):
        """
        PointConv with downsampling.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, S, C ]
            new_points_concat: sample points feature data, [B, S, D']
        """
        #import ipdb; ipdb.set_trace()
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1) # B N C
        points = points.permute(0, 2, 1) # B N D 

        # 降采样至 npoint 个点 (2048)
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = index_points_gather(xyz, fps_idx) # [B, S, C]

        """
        new_points: sampled points position data, [B, S, K, C+D] # [B, npoint, nsample, C+D]
        grouped_xyz_norm: sampled points data, [B, S, K, C] # [B, npoint, nsample, C]
        """
        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, points)

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1) # [B,C,K,S]
        weights = self.weightnet(grouped_xyz) # 通过一系列二维卷积（+BN）变成# [B,16,K,S]
        # B, N, S, C
        # input： B S C+D K ；other B S K 16--> B S C+D 16 --> B S 16*(C+D)
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other = weights.permute(0, 3, 2, 1)).view(B, self.npoint, -1)
        # [B, S, D'] D'= 16 *(64+3)    # self.linear : weightnet * in_channel
        new_points = self.linear(new_points)
        # [B, S, D'] 
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
            new_points = self.relu(new_points)
            new_points = new_points.permute(0, 2, 1)
        else:
            new_points = new_points.permute(0, 2, 1) # [B, D', S]
            new_points = self.relu(new_points)
            new_points = new_points.permute(0, 2, 1)

        # new_xyz:[B, S, C]
        # new_points: [B, S, D'] 
        return new_xyz, new_points, fps_idx
    

# My edit
class PointTransformerLayer(nn.Module):
    """
        PointTransformerLayer
        Input:
            xyz: input points position data, [B, S, C]   # B 8192 3
            points: input points data, [B, S, D']   # B 8192 32
        Return:
            transformered points data : [B*N, out_channel]  对应(n,c)
        """
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=4):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 2
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes))
        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                    nn.Linear(mid_planes, mid_planes // share_planes),
                                    nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
                                    nn.Linear(mid_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, xyz, points) -> torch.Tensor:
        x_q, x_k, x_v = self.linear_q(points), self.linear_k(points), self.linear_v(points) # (B, N, C_mid) (B, N, C_out)

        x_k = new_group_query(self.nsample, xyz, xyz, x_k, use_xyz=True) # [B, npoint, nsample, C+3]
        x_v = new_group_query(self.nsample, xyz, xyz, x_v, use_xyz=False) # [B, npoint, nsample, C]
        
        B, N, nsample, c = x_v.shape # c = C_out
        n = B*N
        s = self.share_planes
        x_q = x_q.contiguous().view(n, self.mid_planes) # (n, C_mid)
        x_k = x_k.contiguous().view(n, nsample, self.mid_planes+3) # (n, nsample, 3+C_mid)
        x_v = x_v.contiguous().view(n, nsample, self.out_planes) # (n, nsample, C_out)

        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        for i, layer in enumerate(self.linear_p): p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)    # p_r:(n, nsample, out_planes)
        w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes, self.mid_planes).sum(2)  # (n, nsample, C_mid)
        for i, layer in enumerate(self.linear_w): w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
        w = self.softmax(w)
        x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
        return x
    
# My edit
class cross_bn_PointTransformerLayer(nn.Module):
    """
        PointTransformerLayer
        Input:
            xyz: input points position data, [B, S, C]   # B 8192 3
            points: input points data, [B, S, D']   # B 8192 32
        Return:
            transformered points data : [B*N, out_channel]  对应(n,c)
        """
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 2
        # self.mid_planes = mid_planes = out_planes
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes))
        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                    nn.Linear(mid_planes, mid_planes // share_planes),
                                    nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
                                    nn.Linear(mid_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)
    
    # 目标：输出  B*N 256     
    def forward(self, xyz1, xyz2, points1, points2) -> torch.Tensor:
        
        x_q, x_k, x_v = self.linear_q(points1), self.linear_k(points2), self.linear_v(points2) # (B, N, C_mid) (B, N, C_out)
        x_k = new_group_query(self.nsample, xyz2, xyz1, x_k, use_xyz=True) # [B, npoint, nsample, C+3]
        x_v = new_group_query(self.nsample, xyz2, xyz1, x_v, use_xyz=False) # [B, npoint, nsample, C]
        
        B, N, nsample, c = x_v.shape # c = C_out
        n = B*N
        s = self.share_planes
        
        x_q = x_q.contiguous().view(n, self.mid_planes) # (n, C_mid)
        x_k = x_k.contiguous().view(n, nsample, self.mid_planes+3) # (n, nsample, 3+C_mid)
        x_v = x_v.contiguous().view(n, nsample, self.out_planes) # (n, nsample, C_out)

        # p_r : [B*N, nsample, C=3]
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        out_p_r = p_r
        for i, layer in enumerate(self.linear_p): p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)    # p_r:(n, nsample, out_planes)
        w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes, self.mid_planes).sum(2)  # (n, nsample, C_mid)
        for i, layer in enumerate(self.linear_w): w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w) # (n, nsample, out_planes // share_planes) 
        w = self.softmax(w)
        # x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
        x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).view(n,nsample, c)
        return x, out_p_r    
    

class cross_PointTransformerLayer(nn.Module):
    """
        PointTransformerLayer
        Input:
            xyz: input points position data, [B, S, C]   # B 8192 3
            points: input points data, [B, S, D']   # B 8192 32
        Return:
            transformered points data : [B*N, out_channel]  对应(n,c)
        """
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        # self.mid_planes = mid_planes = out_planes // 2
        self.mid_planes = mid_planes = out_planes
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        # self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes))
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, out_planes))
        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                    nn.Linear(mid_planes, mid_planes // share_planes),
                                    nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
                                    nn.Linear(mid_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)
    
    # 目标：输出  B*N 256     
    def forward(self, xyz1, xyz2, points1, points2) -> torch.Tensor:
        
        x_q, x_k, x_v = self.linear_q(points1), self.linear_k(points2), self.linear_v(points2) # (B, N, C_mid) (B, N, C_out)
        x_k = new_group_query(self.nsample, xyz2, xyz1, x_k, use_xyz=True) # [B, npoint, nsample, C+3]
        x_v = new_group_query(self.nsample, xyz2, xyz1, x_v, use_xyz=False) # [B, npoint, nsample, C]
        
        B, N, nsample, c = x_v.shape # c = C_out
        n = B*N
        s = self.share_planes
        
        x_q = x_q.contiguous().view(n, self.mid_planes) # (n, C_mid)
        x_k = x_k.contiguous().view(n, nsample, self.mid_planes+3) # (n, nsample, 3+C_mid)
        x_v = x_v.contiguous().view(n, nsample, self.out_planes) # (n, nsample, C_out)

        # p_r : [B*N, nsample, C=3]
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        out_p_r = p_r
        # for i, layer in enumerate(self.linear_p): p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)    # p_r:(n, nsample, out_planes)
        for i, layer in enumerate(self.linear_p): p_r = layer(p_r)
        w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes, self.mid_planes).sum(2)  # (n, nsample, C_mid)
        for i, layer in enumerate(self.linear_w): w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w) # (n, nsample, out_planes // share_planes) 
        w = self.softmax(w)
        x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
        # x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).view(n,nsample, c)   
        # return x
        return x, out_p_r 
          

    
# My edit 
class PointTransformerD(nn.Module):
    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(PointTransformerD, self).__init__()
        self.planes = planes
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)

        self.transformer2 = PointTransformerLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)

        self.linear3 = nn.Linear(planes, planes , bias=False)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, xyz, points):
        """
        PointTransformerD
        Input:
            xyz: input points position data, [B, 3, N]   # B  3 8192
            points: input points data, [B, C, N]   # B  32 8192
        Return:
            transformered points data : [B, out_channel, N]  # B out_channel=64 2048
            输出结果为feat1_l1
            后续还需要外面的 conv1d 变成 feat1_l1_2
        """
        # My edit 
        points = points.transpose(1, 2)  # [B, N ,C]
        xyz = xyz.transpose(1, 2) # [B, N ,3]

        identity = points # B 8192 32
        B, N, _ = points.shape

        points = self.linear1(points) # B 8192 32
        # 当input的维度为（N, C）时，BN将对C维归一化；当input的维度为(N, C, L) 时，归一化的维度同样为C维
        points = self.bn1(points.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        points = self.relu(points)

        # points = self.relu(self.bn1(self.linear1(points))) # B 8192 32
        points = self.relu(self.bn2(self.transformer2(xyz, points))) # B*N 32
        points = self.bn3(self.linear3(points))
        points = points.view(B,N,self.planes)
        # [B, N ,C]
        points += identity
        points = self.relu(points)

        # # origin
        # p, x, o = pxo  # (n, 3), (n, c), (b)
        # identity = x
        # x = self.relu(self.bn1(self.linear1(x)))
        # x = self.relu(self.bn2(self.transformer2([p, x, o])))
        # x = self.bn3(self.linear3(x))
        # x += identity
        # x = self.relu(x)

        # transformered points data : [B, out_channel, N]
        # [B C N]
        return points.transpose(1, 2)




class PointConvK(nn.Module):
    def __init__(self, npoint, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(PointConvK, self).__init__()
        self.npoint = npoint
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.kernel = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.agg = nn.Sequential(
            nn.Conv2d(in_channel, 1, 1, bias=False),
            nn.BatchNorm2d(1),
        )
        self.linear = nn.Linear(out_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz, points):
        """
        PointConv with downsampling.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        #import ipdb; ipdb.set_trace()
        B = xyz.shape[0]
        N = xyz.shape[2]
        C = points.shape[1]
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = index_points_gather(xyz, fps_idx)

        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, points) # [B, npoint, nsample, C+D]
        kernel = self.kernel(new_points.permute(0, 3, 1, 2))
        kernel = self.relu(kernel) # B, Out*In, N, S

        aggregation = torch.matmul(input = kernel.permute(0, 2, 1, 3), other = new_points.permute(0, 1, 2, 3))
        
        aggregation = self.relu(self.agg(aggregation.permute(0, 3, 1, 2))).squeeze(1)
        # B, N, S, C

        new_points = self.linear(aggregation)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_xyz.permute(0, 2, 1), new_points, fps_idx

class SetAbstract(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp, mlp2=None, use_leaky = True):
        super(SetAbstract, self).__init__()
        self.npoint = npoint
        self.nsample = nsample

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        last_channel = in_channel+3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias = False))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        if mlp2:
            for out_channel in mlp2:
                self.mlp2_convs.append(nn.Sequential(nn.Conv1d(last_channel, out_channel, 1, bias=False),
                                                    nn.BatchNorm1d(out_channel)))
                last_channel = out_channel
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        

    def forward(self, xyz, points):
        """
        PointConv with downsampling.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        #import ipdb; ipdb.set_trace()
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = index_points_gather(xyz, fps_idx)

        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, points)

        new_points = new_points.permute(0, 3, 1, 2)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  self.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, -1)[0]

        for i, conv in enumerate(self.mlp2_convs):
            new_points = self.relu(conv(new_points))

        return new_xyz.permute(0, 2, 1), new_points, fps_idx

class PointAtten(nn.Module):
    def __init__(self, npoint, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(PointConvD, self).__init__()
        self.npoint = npoint
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz, points):
        """
        PointConv with downsampling.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        #import ipdb; ipdb.set_trace()
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = index_points_gather(xyz, fps_idx)

        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, points) # B, N, S, C

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz) # B, 16, S, N

        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other = weights.permute(0, 3, 2, 1)).view(B, self.npoint, -1) # B, N, C, S x B, N, S, 16

        new_points = self.linear(new_points)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_xyz.permute(0, 2, 1), new_points, fps_idx

class PointConvFlow(nn.Module):
    def __init__(self, nsample, in_channel, mlp, bn = use_bn, use_leaky = True):
        super(PointConvFlow, self).__init__()
        self.nsample = nsample
        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        if bn:
            self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet1 = WeightNet(3, last_channel)
        self.weightnet2 = WeightNet(3, last_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def forward(self, xyz1, xyz2, points1, points2):
        """
        Cost Volume layer for Flow Estimation
        Input:
            xyz1: input points position data, [B, C, N1]
            xyz2: input points position data, [B, C, N2]
            points1: input points data, [B, D, N1]
            points2: input points data, [B, D, N2]
        Return:
            new_points: upsample points feature data, [B, D', N1]
        """
        # import ipdb; ipdb.set_trace()
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        # point-to-patch Volume
        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        grouped_points2 = index_points_group(points2, knn_idx) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1)
        new_points = torch.cat([grouped_points1, grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3
        new_points = new_points.permute(0, 3, 2, 1) # [B, D1+D2+3, nsample, N1]
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                new_points =  self.relu(bn(conv(new_points)))
            else:
                new_points =  self.relu(conv(new_points))

        # weighted sum
        weights = self.weightnet1(direction_xyz.permute(0, 3, 2, 1)) # B C nsample N1 

        point_to_patch_cost = torch.sum(weights * new_points, dim = 2) # B C N

        # Patch to Patch Cost
        knn_idx = knn_point(self.nsample, xyz1, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz1, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        # weights for group cost
        weights = self.weightnet2(direction_xyz.permute(0, 3, 2, 1)) # B C nsample N1 
        grouped_point_to_patch_cost = index_points_group(point_to_patch_cost.permute(0, 2, 1), knn_idx) # B, N1, nsample, C
        patch_to_patch_cost = torch.sum(weights * grouped_point_to_patch_cost.permute(0, 3, 2, 1), dim = 2) # B C N

        return patch_to_patch_cost


class CrossLayer(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossLayer,self).__init__()
        # self.fe1_layer = FlowEmbedding(radius=radius, nsample=nsample, in_channel = in_channel, mlp=[in_channel,in_channel], pooling=pooling, corr_func=corr_func)
        # self.fe2_layer = FlowEmbedding(radius=radius, nsample=nsample, in_channel = in_channel, mlp=[in_channel, out_channel], pooling=pooling, corr_func=corr_func)
        # self.flow = nn.Conv1d(out_channel, 3, 1)

        self.nsample = nsample
        self.bn = bn
        self.mlp1_convs = nn.ModuleList()
        if bn:
            self.mlp1_bns = nn.ModuleList()
        last_channel = in_channel  * 2 + 3
        for out_channel in mlp1:
            self.mlp1_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp1_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        if mlp2 is not None:
            self.mlp2_convs = nn.ModuleList()
            if bn:
                self.mlp2_bns = nn.ModuleList()
            last_channel = mlp1[-1] * 2 + 3
            for out_channel in mlp2:
                self.mlp2_convs.append(nn.Conv2d(last_channel, out_channel, 1))
                if bn:
                    self.mlp2_bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def cross(self, xyz1, xyz2, points1, points2, mlp_convs, mlp_bns):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        grouped_points2 = index_points_group(points2, knn_idx) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1)
        new_points = torch.cat([grouped_points1, grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3
        new_points = new_points.permute(0, 3, 2, 1) # [B, D1+D2+3, nsample, N1]
        for i, conv in enumerate(mlp_convs):
            if self.bn:
                bn = mlp_bns[i]
                new_points =  self.relu(bn(conv(new_points)))
            else:
                new_points =  self.relu(conv(new_points))
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points
    
    def forward(self, pc1, pc2, feat1, feat2):
        # _, feat1_new = self.fe1_layer(pc1, pc2, feat1, feat2)
        # _, feat2_new = self.fe1_layer(pc2, pc1, feat2, feat1)
        # _, feat1_final = self.fe2_layer(pc1, pc2, feat1_new, feat2_new)
        # flow1 = self.flow(feat1_final)

        feat1_new = self.cross(pc1, pc2, feat1, feat2, self.mlp1_convs, self.mlp1_bns if self.bn else None)
        feat2_new = self.cross(pc2, pc1, feat2, feat1, self.mlp1_convs, self.mlp1_bns if self.bn else None)
        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.mlp2_convs, self.mlp2_bns if self.bn else None)

        return feat1_new, feat2_new, feat1_final

class CrossLayerLight(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossLayerLight,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        last_channel = in_channel

        # Round 1
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        # self.cross_t12 = Conv1d(in_channel, mlp1[0], bn=bn, use_leaky=use_leaky)
        # self.cross_t21 = Conv1d(in_channel, mlp1[0], bn=bn, use_leaky=use_leaky)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()
        
            
        self.mlp1 = nn.ModuleList()
        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]
        

        # Round 2
        self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
        self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)

        self.pos2 = nn.Conv2d(3, mlp2[0], 1)
        self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
        self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

        self.mlp2 = nn.ModuleList()
        for i in range(1, len(mlp2)):
            self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def cross(self, xyz1, xyz2, points1, points2, pos, mlp, bn):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1) # B 256 3
        xyz2 = xyz2.permute(0, 2, 1) # B 256 3
        points1 = points1.permute(0, 2, 1) # B N=256 256  //256+64过cross_t11变256//
        points2 = points2.permute(0, 2, 1) # B N=256 256  //256+64过cross_t11变256//

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample,找距离xyz1 中点最近的32个xyz2点
        # direction_xyz: [B, N, K=32, C=3 即坐标]
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        # direction_xyz 为32个KNN 到中心点的方向向量[B, N, K=32, C=3 即坐标]
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        # [B, N, K=32, C=256 即特征]
        # permute 后 grouped_points2 :[ B C=256 K=16 N=256]
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        # (B, N1, K=32, D1)
        # permute 后 grouped_points1 :[ B D1=256 K=16 N1=256]
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)


        # direction_xyz.permute(0, 3, 2, 1) :[B, C=3 即坐标, K=16, N]
        # 经过pos: conv2d(3,256,1) : direction_xyz [B 256 16 N=256]
        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        # [B 256 16 N=256]
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        # [B 256 16 N=256]
        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        

        # max_pool2d(new_points, (new_points.size(2), 1)) ---> [B 256 16 N=256]
        # [B 256 N=256]
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points
    
    def forward(self, pc1, pc2, feat1, feat2):
        # _, feat1_new = self.fe1_layer(pc1, pc2, feat1, feat2)
        # _, feat2_new = self.fe1_layer(pc2, pc1, feat2, feat1)
        # _, feat1_final = self.fe2_layer(pc1, pc2, feat1_new, feat2_new)
        # flow1 = self.flow(feat1_final)


        # 找距离xyz1 中点最近的16个xyz2点的特征，自己特征复制K=16份，相对坐标3维通过conv2d提升到256维度，三个相加后过conv2d + 对K这维度的maxpool
        # pc1 pc2 :[B C S]
        feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1), self.cross_t22(feat2), self.pos1, self.mlp1, self.bn1)
        feat1_new = self.cross_t1(feat1_new)
        feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.pos1, self.mlp1, self.bn1)
        feat2_new = self.cross_t2(feat2_new)

        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.pos2, self.mlp2, self.bn2)

        return feat1_new, feat2_new, feat1_final


class knn_CrossLayerLight(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(knn_CrossLayerLight,self).__init__()

        self.truncate_k = 512
        self.weightnet = KNN_WeightNet(4, nsample)

        self.nsample = nsample
        self.bn = bn
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        last_channel = in_channel

        # Round 1
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        # self.cross_t12 = Conv1d(in_channel, mlp1[0], bn=bn, use_leaky=use_leaky)
        # self.cross_t21 = Conv1d(in_channel, mlp1[0], bn=bn, use_leaky=use_leaky)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()
        
            
        self.mlp1 = nn.ModuleList()
        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]
        

        # Round 2
        self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
        self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)

        self.pos2 = nn.Conv2d(3, mlp2[0], 1)
        self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
        self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

        self.mlp2 = nn.ModuleList()
        for i in range(1, len(mlp2)):
            self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    @staticmethod
    def calculate_corr(fmap1, fmap2):
        batch, dim, num_points = fmap1.shape
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr / torch.sqrt(torch.tensor(dim).float())
        return corr
    
    def cross(self, xyz1, xyz2, points1, points2, pos, mlp, bn, corr_topk):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape


        xyz1 = xyz1.permute(0, 2, 1) # B 256 3
        xyz2 = xyz2.permute(0, 2, 1) # B 256 3
        points1 = points1.permute(0, 2, 1) # B N=256 256  //256+64过cross_t11变256//
        points2 = points2.permute(0, 2, 1) # B N=256 256  //256+64过cross_t11变256//

        truncated_corr = corr_topk.values # B N K
        indx = corr_topk.indices # B N K
        _, _, K = indx.shape

        valid_xyz = index_points_group(xyz2, indx) # [B, N, K=truncate_k, C=3 即坐标]
        valid_xyz = valid_xyz - xyz1.view(B, N1, 1, C)
        input_knn_feature = torch.cat([valid_xyz, truncated_corr.reshape(B, N1, K, 1)], dim = -1) # B, N1, K=truncate_k, 4

        knn_idx = self.weightnet(input_knn_feature)

        # knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample,找距离xyz1 中点最近的32个xyz2点

        # direction_xyz: [B, N, K=32, C=3 即坐标]
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        # direction_xyz 为32个KNN 到中心点的方向向量[B, N, K=32, C=3 即坐标]
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        # [B, N, K=32, C=256 即特征]
        # permute 后 grouped_points2 :[ B C=256 K=16 N=256]
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        # (B, N1, K=32, D1)
        # permute 后 grouped_points1 :[ B D1=256 K=16 N1=256]
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)


        # direction_xyz.permute(0, 3, 2, 1) :[B, C=3 即坐标, K=16, N]
        # 经过pos: conv2d(3,256,1) : direction_xyz [B 256 16 N=256]
        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        # [B 256 16 N=256]
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        # [B 256 16 N=256]
        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        

        # max_pool2d(new_points, (new_points.size(2), 1)) ---> [B 256 16 N=256]
        # [B 256 N=256]
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points
    
    def forward(self, pc1, pc2, feat1, feat2):
        # _, feat1_new = self.fe1_layer(pc1, pc2, feat1, feat2)
        # _, feat2_new = self.fe1_layer(pc2, pc1, feat2, feat1)
        # _, feat1_final = self.fe2_layer(pc1, pc2, feat1_new, feat2_new)
        # flow1 = self.flow(feat1_final)


        # 找距离xyz1 中点最近的16个xyz2点的特征，自己特征复制K=16份，相对坐标3维通过conv2d提升到256维度，三个相加后过conv2d + 对K这维度的maxpool
        # pc1 pc2 :[B C S]

        # Round 1
        fmap1 = self.cross_t11(feat1) # B D1=256 N=256  //256+64过cross_t11变256//
        fmap2 = self.cross_t22(feat2) # B D1=256 N=256  //256+64过cross_t11变256//
        B, D1, N = fmap1.shape
        truncate_k = min(self.truncate_k, N)
        corr = self.calculate_corr(fmap1, fmap2)
        corr_topk = torch.topk(corr.clone(), k=truncate_k, dim=2, sorted=True) # B N K=truncate_k

        feat1_new = self.cross(pc1, pc2, fmap1, fmap2, self.pos1, self.mlp1, self.bn1, corr_topk)
        feat1_new = self.cross_t1(feat1_new)

        # Round 2
        fmap1_R2 = self.cross_t11(feat2)
        fmap2_R2 = self.cross_t22(feat1)
        corr_R2 = self.calculate_corr(fmap1_R2, fmap2_R2)
        corr_topk_R2 = torch.topk(corr_R2.clone(), k=truncate_k, dim=2, sorted=True) # B N K=truncate_k

        feat2_new = self.cross(pc2, pc1, fmap1_R2, fmap2_R2, self.pos1, self.mlp1, self.bn1,corr_topk_R2)
        feat2_new = self.cross_t2(feat2_new)

        # Round 3
        corr_R3 = self.calculate_corr(feat1_new, feat2_new)
        corr_topk_R3 = torch.topk(corr_R3.clone(), k=truncate_k, dim=2, sorted=True) # B N K=truncate_k

        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.pos2, self.mlp2, self.bn2,corr_topk_R3)

        # feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1), self.cross_t22(feat2), self.pos1, self.mlp1, self.bn1, corr_topk)
        # feat1_new = self.cross_t1(feat1_new)
        # feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.pos1, self.mlp1, self.bn1)
        # feat2_new = self.cross_t2(feat2_new)

        # feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.pos2, self.mlp2, self.bn2)

        return feat1_new, feat2_new, feat1_final
    



class knn_v2_CrossLayerLight(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(knn_v2_CrossLayerLight,self).__init__()

        self.truncate_k = 512
        # self.weightnet = KNN_WeightNet(4, nsample)

        self.nsample = nsample
        self.bn = bn
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        last_channel = in_channel

        # Round 1
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        # self.cross_t12 = Conv1d(in_channel, mlp1[0], bn=bn, use_leaky=use_leaky)
        # self.cross_t21 = Conv1d(in_channel, mlp1[0], bn=bn, use_leaky=use_leaky)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()
        
            
        self.mlp1 = nn.ModuleList()
        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]
        

        # Round 2
        self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
        self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)

        self.pos2 = nn.Conv2d(3, mlp2[0], 1)
        self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
        self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

        self.mlp2 = nn.ModuleList()
        for i in range(1, len(mlp2)):
            self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    @staticmethod
    def calculate_corr(fmap1, fmap2):
        batch, dim, num_points = fmap1.shape
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr / torch.sqrt(torch.tensor(dim).float())
        return corr
    
    def cross(self, xyz1, xyz2, points1, points2, pos, mlp, bn, corr_topk):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape


        xyz1 = xyz1.permute(0, 2, 1) # B 256 3
        xyz2 = xyz2.permute(0, 2, 1) # B 256 3
        points1 = points1.permute(0, 2, 1) # B N=256 256  //256+64过cross_t11变256//
        points2 = points2.permute(0, 2, 1) # B N=256 256  //256+64过cross_t11变256//

        # truncated_corr = corr_topk.values # B N K
        indx = corr_topk.indices # B N K
        # _, _, K = indx.shape

        valid_xyz = index_points_group(xyz2, indx) # [B, N, K=truncate_k, C=3 即坐标]
        dist = valid_xyz - xyz1.view(B, N1, 1, C)
        dist = torch.sum(dist ** 2, dim=-1)
        knn_idx = torch.topk(-dist, k=self.nsample, dim=2).indices
        
        
        # input_knn_feature = torch.cat([valid_xyz, truncated_corr.reshape(B, N1, K, 1)], dim = -1) # B, N1, K=truncate_k, 4
        # knn_idx = self.weightnet(input_knn_feature)
        # knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample,找距离xyz1 中点最近的32个xyz2点

        # direction_xyz: [B, N, K=32, C=3 即坐标]
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        # direction_xyz 为32个KNN 到中心点的方向向量[B, N, K=32, C=3 即坐标]
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        # [B, N, K=32, C=256 即特征]
        # permute 后 grouped_points2 :[ B C=256 K=16 N=256]
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        # (B, N1, K=32, D1)
        # permute 后 grouped_points1 :[ B D1=256 K=16 N1=256]
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)


        # direction_xyz.permute(0, 3, 2, 1) :[B, C=3 即坐标, K=16, N]
        # 经过pos: conv2d(3,256,1) : direction_xyz [B 256 16 N=256]
        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        # [B 256 16 N=256]
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        # [B 256 16 N=256]
        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        

        # max_pool2d(new_points, (new_points.size(2), 1)) ---> [B 256 16 N=256]
        # [B 256 N=256]
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points
    
    def forward(self, pc1, pc2, feat1, feat2):
        # _, feat1_new = self.fe1_layer(pc1, pc2, feat1, feat2)
        # _, feat2_new = self.fe1_layer(pc2, pc1, feat2, feat1)
        # _, feat1_final = self.fe2_layer(pc1, pc2, feat1_new, feat2_new)
        # flow1 = self.flow(feat1_final)


        # 找距离xyz1 中点最近的16个xyz2点的特征，自己特征复制K=16份，相对坐标3维通过conv2d提升到256维度，三个相加后过conv2d + 对K这维度的maxpool
        # pc1 pc2 :[B C S]

        # Round 1
        fmap1 = self.cross_t11(feat1) # B D1=256 N=256  //256+64过cross_t11变256//
        fmap2 = self.cross_t22(feat2) # B D1=256 N=256  //256+64过cross_t11变256//
        B, D1, N = fmap1.shape
        truncate_k = min(self.truncate_k, N)
        corr = self.calculate_corr(fmap1, fmap2)
        corr_topk = torch.topk(corr.clone(), k=truncate_k, dim=2, sorted=True) # B N K=truncate_k

        feat1_new = self.cross(pc1, pc2, fmap1, fmap2, self.pos1, self.mlp1, self.bn1, corr_topk)
        feat1_new = self.cross_t1(feat1_new)

        # Round 2
        fmap1_R2 = self.cross_t11(feat2)
        fmap2_R2 = self.cross_t22(feat1)
        corr_R2 = self.calculate_corr(fmap1_R2, fmap2_R2)
        corr_topk_R2 = torch.topk(corr_R2.clone(), k=truncate_k, dim=2, sorted=True) # B N K=truncate_k

        feat2_new = self.cross(pc2, pc1, fmap1_R2, fmap2_R2, self.pos1, self.mlp1, self.bn1,corr_topk_R2)
        feat2_new = self.cross_t2(feat2_new)

        # Round 3
        corr_R3 = self.calculate_corr(feat1_new, feat2_new)
        corr_topk_R3 = torch.topk(corr_R3.clone(), k=truncate_k, dim=2, sorted=True) # B N K=truncate_k

        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.pos2, self.mlp2, self.bn2,corr_topk_R3)

        # feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1), self.cross_t22(feat2), self.pos1, self.mlp1, self.bn1, corr_topk)
        # feat1_new = self.cross_t1(feat1_new)
        # feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.pos1, self.mlp1, self.bn1)
        # feat2_new = self.cross_t2(feat2_new)

        # feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.pos2, self.mlp2, self.bn2)

        return feat1_new, feat2_new, feat1_final
    



class pt_bn_CrossLayerLight(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True, share_planes=8):
        super(pt_bn_CrossLayerLight,self).__init__()

        self.nsample = nsample
        self.bn = bn
        mid_planes = mlp1[0]
        self.pt_nsample = 16
        
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bn1 = nn.BatchNorm1d(mlp1[0])
        self.bn2 = nn.BatchNorm1d(mlp2[0])
        self.bn_old = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        self.pos2 = nn.Conv2d(3, mlp2[0], 1)
        self.relu = nn.ReLU(inplace=True)
        
        # 目标：输出  B*N 256      [ B D1=256 K=16 N1=256]
        # self.transformer(xyz1, xyz2, points1, points2) # B*N 256 
        self.transformer = cross_bn_PointTransformerLayer(mid_planes, mid_planes, share_planes, self.pt_nsample)
        
        self.bn_t1 = nn.BatchNorm1d(mid_planes) 
        # self.bn_t2 = nn.BatchNorm1d(mid_planes)
        
        self.bn_t11 = nn.BatchNorm2d(mid_planes) if bn else nn.Identity()
        # self.bn_t22 = nn.BatchNorm1d(mid_planes)
        
        self.mlp1 = nn.ModuleList()
        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))

        
        # self.cross_t1 = nn.Conv1d(2 * mlp1[0], mlp1[1], 1)
        # self.cross_t2 = nn.Conv1d(2 * mlp2[0], mlp2[1], 1)
        
        self.bn_old = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()
        self.pos_old = nn.Conv2d(3, mlp2[0], 1)
        self.mlp_old = nn.ModuleList()
        for i in range(1, len(mlp2)):
            self.mlp_old.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))
 
 

    def cross(self, xyz1, xyz2, points1, points2, bn1, bn2, pos, mlp):
        # My Edit
        # xyz1 B N=256 3   [B, N1 ,3]
        # xyz2 B N=256 3   [B, N2 ,3]
        B, N1, C = xyz1.shape
        _, N2, _ = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        
        points1 = self.bn1(points1)
        points1 = points1.permute(0, 2, 1) # [B, N ,D1]
        points1 = self.relu(points1)
        
        points2 = self.bn2(points2)
        points2 = points2.permute(0, 2, 1) # [B, N ,D1]
        points2 = self.relu(points2)
        
        # n,nsample, c
        grouped_points2, direction_xyz = self.transformer(xyz1, xyz2, points1, points2)
        direction_xyz = direction_xyz.view(B, N1, self.pt_nsample, C)
        # 经过pos: conv2d(3,256,1) : direction_xyz [B 256 16 N=256]
        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
               
        # new_points = self.relu(bn(new_points))
        # new_points = new_points.view(B,N1, D1) # B N 256
        # new_points = torch.cat([points1, new_points], dim = 2) # B N 512 
        # new_points = cross_cov(new_points.permute(0, 2, 1)) # B 256 N
        
        grouped_points2 = self.relu(bn1(grouped_points2.permute(0, 2, 1)).permute(0, 2, 1)) # B*N 256       
        grouped_points2 = grouped_points2.view(B,N1,self.pt_nsample, D1) # B N1 16 256
        # permute 后 grouped_points2 :[ B C=256 K=16 N=256]
        grouped_points2 = grouped_points2.permute(0, 3, 2, 1)
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.pt_nsample, 1).permute(0, 3, 2, 1)
        
        new_points = self.relu(bn2(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)
        
        return new_points
    
    def old_cross(self, xyz1, xyz2, points1, points2, pos, mlp, bn):
        B, N1, C = xyz1.shape
        _, N2, _ = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        # xyz1 = xyz1.permute(0, 2, 1) # B 256 3
        # xyz2 = xyz2.permute(0, 2, 1) # B 256 3
        points1 = points1.permute(0, 2, 1) # [B, N ,D1] //256+64过cross_t11变256//
        points2 = points2.permute(0, 2, 1) # [B, N ,D2] //256+64过cross_t11变256//

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample,找距离xyz1 中点最近的32个xyz2点
        # direction_xyz: [B, N, K=32, C=3 即坐标]
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        # direction_xyz 为32个KNN 到中心点的方向向量[B, N, K=32, C=3 即坐标]
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        # [B, N, K=32, C=256 即特征]
        # permute 后 grouped_points2 :[ B C=256 K=16 N=256]
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        # (B, N1, K=32, D1)
        # permute 后 grouped_points1 :[ B D1=256 K=16 N1=256]
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)


        # direction_xyz.permute(0, 3, 2, 1) :[B, C=3 即坐标, K=16, N]
        # 经过pos: conv2d(3,256,1) : direction_xyz [B 256 16 N=256]
        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        # [B 256 16 N=256]
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, D1+D2+3, nsample, N1

        # [B 256 16 N=256]
        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        

        # max_pool2d(new_points, (new_points.size(2), 1)) ---> [B 256 16 N=256]
        # [B 256 N=256]
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points


    def forward(self, pc1, pc2, feat1, feat2):
        # _, feat1_new = self.fe1_layer(pc1, pc2, feat1, feat2)
        # _, feat2_new = self.fe1_layer(pc2, pc1, feat2, feat1)
        # _, feat1_final = self.fe2_layer(pc1, pc2, feat1_new, feat2_new)
        # flow1 = self.flow(feat1_final)
        # pc1 pc2 :[B C S]  -->  [B S=256 3]
        pc1 = pc1.permute(0, 2, 1)
        pc2 = pc2.permute(0, 2, 1)

        feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1), self.cross_t22(feat2), self.bn_t1,self.bn_t11, self.pos1, self.mlp1)
        # feat1_new = self.cross_t1(feat1_new)
        feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.bn_t1,self.bn_t11, self.pos2, self.mlp1)
        # feat2_new = self.cross_t2(feat2_new)

        feat1_final = self.old_cross(pc1, pc2, feat1_new, feat2_new, self.pos_old, self.mlp_old, self.bn_old)

        return feat1_new, feat2_new, feat1_final



class pt_CrossLayerLight(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True, share_planes=8):
        super(pt_CrossLayerLight,self).__init__()

        self.nsample = nsample
        self.bn = bn
        mid_planes = mlp1[0]
        self.pt_nsample = 32
        
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        # self.bn1 = nn.BatchNorm1d(mlp1[0])
        # self.bn2 = nn.BatchNorm1d(mlp2[0])
        self.bn_old = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        self.pos2 = nn.Conv2d(3, mlp2[0], 1)
        self.relu = nn.ReLU(inplace=True)
        
        # 目标：输出  B*N 256      [ B D1=256 K=16 N1=256]
        # self.transformer(xyz1, xyz2, points1, points2) # B*N 256 
        self.transformer = cross_PointTransformerLayer(mid_planes, mid_planes, share_planes, self.pt_nsample)
        
        # self.bn_t1 = nn.BatchNorm1d(mid_planes) 
        # self.bn_t2 = nn.BatchNorm1d(mid_planes)
        
        # self.bn_t11 = nn.BatchNorm2d(mid_planes) if bn else nn.Identity()
        # self.bn_t22 = nn.BatchNorm1d(mid_planes)
        
        self.mlp1 = nn.ModuleList()
        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
        
        # self.cross_c1 = nn.Conv1d(2 * mlp1[0], mlp1[1], 1)
        # self.cross_c11 = nn.Conv1d( mlp1[1], mlp1[1], 1)
        # self.cross_c2 = nn.Conv1d(2 * mlp2[0], mlp2[1], 1)
        # self.cross_c22 = nn.Conv1d( mlp2[1], mlp2[1], 1)
        
        self.cross_final1 = nn.Conv1d(2 * mlp2[0], mlp2[1], 1)
        self.cross_final2 = nn.Conv1d( mlp2[1], mlp2[1], 1)
        
        self.bn_old = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()
        self.pos_old = nn.Conv2d(3, mlp2[0], 1)
        self.mlp_old = nn.ModuleList()
        for i in range(1, len(mlp2)):
            self.mlp_old.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))


    def cross(self, xyz1, xyz2, points1, points2, pos, mlp):
        # My Edit
        # xyz1 B N=256 3   [B, N1 ,3]
        # xyz2 B N=256 3   [B, N2 ,3]
        B, N1, C = xyz1.shape
        _, N2, _ = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
               
        points1 = points1.permute(0, 2, 1) # [B, N ,D1]
        points2 = points2.permute(0, 2, 1) # [B, N ,D1]
        
        # n,nsample, c
        grouped_points2, direction_xyz = self.transformer(xyz1, xyz2, points1, points2)
        # grouped_points2 = self.transformer(xyz1, xyz2, points1, points2).view(B ,N1, D2)
        direction_xyz = direction_xyz.view(B, N1, self.pt_nsample, C)
        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.pt_nsample, 1).permute(0, 3, 2, 1)
        grouped_points2 = grouped_points2.view(B, N1, 1, D1).repeat(1, 1, self.pt_nsample, 1).permute(0, 3, 2, 1)
        new_points = self.relu(grouped_points2 + grouped_points1 + direction_xyz)# B, N1, nsample, D1+D2+3
        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)
        
        # new_points = torch.cat([points1, grouped_points2], dim = 2) # B N1  D1+D2 = 512 
        # new_points = self.relu(cross_conv1(new_points.permute(0, 2, 1))) # B 256 N
        # new_points = self.relu(cross_conv2(new_points))

        return new_points
 
 

    def v3_cross(self, xyz1, xyz2, points1, points2, bn2, pos, mlp):
        # My Edit
        # xyz1 B N=256 3   [B, N1 ,3]
        # xyz2 B N=256 3   [B, N2 ,3]
        B, N1, C = xyz1.shape
        _, N2, _ = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        
        # origin_points1 = points1.permute(0, 2, 1) # [B, N ,D1]
        # origin_points2 = points2.permute(0, 2, 1) # [B, N ,D1]
        
        # points1 = self.bn1(points1)
        # points1 = points1.permute(0, 2, 1) # [B, N ,D1]
        # points1 = self.relu(points1)
        
        # points2 = self.bn2(points2)
        # points2 = points2.permute(0, 2, 1) # [B, N ,D1]
        # points2 = self.relu(points2)
        
        points1 = points1.permute(0, 2, 1) # [B, N ,D1]
        points2 = points2.permute(0, 2, 1) # [B, N ,D1]
        
        # n,nsample, c
        grouped_points2, direction_xyz = self.transformer(xyz1, xyz2, points1, points2)
        direction_xyz = direction_xyz.view(B, N1, self.pt_nsample, C)
        # 经过pos: conv2d(3,256,1) : direction_xyz [B 256 16 N=256]
        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
               
        # new_points = self.relu(bn(new_points))
        # new_points = new_points.view(B,N1, D1) # B N 256
        # new_points = torch.cat([points1, new_points], dim = 2) # B N 512 
        # new_points = cross_cov(new_points.permute(0, 2, 1)) # B 256 N
        
        # grouped_points2 = self.relu(bn1(grouped_points2.permute(0, 2, 1)).permute(0, 2, 1)) # B*N 256  
        grouped_points2 = grouped_points2.view(B,N1,self.pt_nsample, D1) # B N1 16 256
        # permute 后 grouped_points2 :[ B C=256 K=16 N=256]
        grouped_points2 = grouped_points2.permute(0, 3, 2, 1)
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.pt_nsample, 1).permute(0, 3, 2, 1)
        
        new_points = self.relu(bn2(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)
        
        return new_points
    
    def old_cross(self, xyz1, xyz2, points1, points2, pos, mlp, bn):
        B, N1, C = xyz1.shape
        _, N2, _ = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        # xyz1 = xyz1.permute(0, 2, 1) # B 256 3
        # xyz2 = xyz2.permute(0, 2, 1) # B 256 3
        points1 = points1.permute(0, 2, 1) # [B, N ,D1] //256+64过cross_t11变256//
        points2 = points2.permute(0, 2, 1) # [B, N ,D2] //256+64过cross_t11变256//

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample,找距离xyz1 中点最近的32个xyz2点
        # direction_xyz: [B, N, K=32, C=3 即坐标]
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        # direction_xyz 为32个KNN 到中心点的方向向量[B, N, K=32, C=3 即坐标]
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        # [B, N, K=32, C=256 即特征]
        # permute 后 grouped_points2 :[ B C=256 K=16 N=256]
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        # (B, N1, K=32, D1)
        # permute 后 grouped_points1 :[ B D1=256 K=16 N1=256]
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)


        # direction_xyz.permute(0, 3, 2, 1) :[B, C=3 即坐标, K=16, N]
        # 经过pos: conv2d(3,256,1) : direction_xyz [B 256 16 N=256]
        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        # [B 256 16 N=256]
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, D1+D2+3, nsample, N1

        # [B 256 16 N=256]
        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        

        # max_pool2d(new_points, (new_points.size(2), 1)) ---> [B 256 16 N=256]
        # [B 256 N=256]
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points


    def forward(self, pc1, pc2, feat1, feat2):
        # _, feat1_new = self.fe1_layer(pc1, pc2, feat1, feat2)
        # _, feat2_new = self.fe1_layer(pc2, pc1, feat2, feat1)
        # _, feat1_final = self.fe2_layer(pc1, pc2, feat1_new, feat2_new)
        # flow1 = self.flow(feat1_final)
        # pc1 pc2 :[B C S]  -->  [B S=256 3]
        pc1 = pc1.permute(0, 2, 1)
        pc2 = pc2.permute(0, 2, 1)

        # v3
        # feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1), self.cross_t22(feat2),self.bn_t11, self.pos1, self.mlp1)
        # feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1),self.bn_t11, self.pos2, self.mlp1)

        # v4
        feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1), self.cross_t22(feat2), self.pos1, self.mlp1)
        feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.pos2, self.mlp1)
        
        # feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new,self.cross_final1,self.cross_final2)

        feat1_final = self.old_cross(pc1, pc2, feat1_new, feat2_new, self.pos_old, self.mlp_old, self.bn_old)

        return feat1_new, feat2_new, feat1_final



class PointWarping(nn.Module):

    def forward(self, xyz1, xyz2, flow1 = None):
        if flow1 is None:
            return xyz2

        # move xyz1 to xyz2'
        xyz1_to_2 = xyz1 + flow1 

        # interpolate flow
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        xyz1_to_2 = xyz1_to_2.permute(0, 2, 1) # B 3 N1
        xyz2 = xyz2.permute(0, 2, 1) # B 3 N2
        flow1 = flow1.permute(0, 2, 1)

        # 3 nearest neightbor & use 1/dist as the weights
        knn_idx = knn_point(3, xyz1_to_2, xyz2) # group flow 1 around points 2
        grouped_xyz_norm = index_points_group(xyz1_to_2, knn_idx) - xyz2.view(B, N2, 1, C) # B N2 3 C
        dist = torch.norm(grouped_xyz_norm, dim = 3).clamp(min = 1e-10) 
        norm = torch.sum(1.0 / dist, dim = 2, keepdim = True) 
        weight = (1.0 / dist) / norm 

        # from points 2 to group flow 1 and got weight, and use these weights and grouped flow to wrap a inverse flow and flow back
        grouped_flow1 = index_points_group(flow1, knn_idx)
        flow2 = torch.sum(weight.view(B, N2, 3, 1) * grouped_flow1, dim = 2)
        warped_xyz2 = (xyz2 - flow2).permute(0, 2, 1) # B 3 N2

        return warped_xyz2

class UpsampleFlow(nn.Module):
    def forward(self, xyz, sparse_xyz, sparse_flow):
        #import ipdb; ipdb.set_trace()
        B, C, N = xyz.shape
        _, _, S = sparse_xyz.shape

        xyz = xyz.permute(0, 2, 1) # B N 3
        sparse_xyz = sparse_xyz.permute(0, 2, 1) # B S 3
        sparse_flow = sparse_flow.permute(0, 2, 1) # B S 3

        # offset = np.linspace(N,B*N,B, dtype = int)
        # offset = torch.cuda.IntTensor(offset)

        # new_offset = np.linspace(S,B*S,B, dtype = int)
        # new_offset = torch.cuda.IntTensor(new_offset)

        # 3 nearest neightbor from dense around sparse & use 1/dist as the weights the same
        #  [B, S, 3]
        # 找xyz 附近的 3个sparse_xyz
        knn_idx = knn_point(3, sparse_xyz, xyz)

        # [B, N, K=3, C=3 即坐标]
        # xyz 附近的 3个sparse_xyz 到中心点的距离
        grouped_xyz_norm = index_points_group(sparse_xyz, knn_idx) - xyz.view(B, N, 1, C)
        # dist = [B, N, K=3] 
        dist = torch.norm(grouped_xyz_norm, dim = 3).clamp(min = 1e-10)
        norm = torch.sum(1.0 / dist, dim = 2, keepdim = True)
        #  xyz 附近的 3个sparse_xyz 到中心点的距离倒数归一化
        # weight = [B, N, K=3] 
        weight = (1.0 / dist) / norm 

        # [B, N, K=3, C=256 即特征sparse_flow]
        # B N 3 1 * B N 3 256
        # sum: B N 256
        # dense_flow B 256 N ,N = xyz
        grouped_flow = index_points_group(sparse_flow, knn_idx)
        dense_flow = torch.sum(weight.view(B, N, 3, 1) * grouped_flow, dim = 2).permute(0, 2, 1)
        return dense_flow 

class SceneFlowEstimatorPointConv(nn.Module):

    def __init__(self, feat_ch, cost_ch, flow_ch = 3, channels = [128, 128], mlp = [128, 64], neighbors = 9, clamp = [-200, 200], use_leaky = True):
        super(SceneFlowEstimatorPointConv, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch + cost_ch + flow_ch

        for _, ch_out in enumerate(channels):
            pointconv = PointConv(neighbors, last_channel + 3, ch_out, bn = True, use_leaky = True)
            self.pointconv_list.append(pointconv)
            last_channel = ch_out 
        
        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 3, 1)

    def forward(self, xyz, feats, cost_volume, flow = None):
        '''
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        '''
        if flow is None:
            new_points = torch.cat([feats, cost_volume], dim = 1)
        else:
            new_points = torch.cat([feats, cost_volume, flow], dim = 1)

        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv(xyz, new_points)

        for conv in self.mlp_convs:
            new_points = conv(new_points)

        flow = self.fc(new_points)
        return new_points, flow.clamp(self.clamp[0], self.clamp[1])

class SceneFlowEstimatorResidual(nn.Module):

    def __init__(self, feat_ch, cost_ch, flow_ch = 3, channels = [128, 128], mlp = [128, 64], neighbors = 9, clamp = [-200, 200], use_leaky = True):
        super(SceneFlowEstimatorResidual, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch + cost_ch

        for _, ch_out in enumerate(channels):
            pointconv = PointConv(neighbors, last_channel + 3, ch_out, bn = True, use_leaky = True)
            self.pointconv_list.append(pointconv)
            last_channel = ch_out 
        
        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 3, 1)

    def forward(self, xyz, feats, cost_volume, flow = None):
        '''
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        '''
        # B 256+256=512 256
        new_points = torch.cat([feats, cost_volume], dim = 1)

        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv(xyz, new_points)

        for conv in self.mlp_convs:
            new_points = conv(new_points)

        flow_local = self.fc(new_points).clamp(self.clamp[0], self.clamp[1]) 
        
        if flow is None:
            flow = flow_local
        else:
            flow = flow_local + flow
        return new_points, flow


# My edit 
class pt_SceneFlowEstimatorResidual(nn.Module):
    def __init__(self, feat_ch, cost_ch, neighbors = 9,channels = [128, 128], mlp = [128, 64], mid_planes = 128, planes = 64, share_planes=8 , nsample = 16, clamp = [-200, 200], use_leaky = True):
    # def __init__(self, feat_ch, cost_ch, flow_ch = 3, channels = [128, 128], mlp = [128, 64], neighbors = 9, clamp = [-200, 200], use_leaky = True):
        super(pt_SceneFlowEstimatorResidual, self).__init__()
        self.use_leaky = use_leaky
        self.clamp = clamp
        self.mid_planes = mid_planes
        last_channel = feat_ch + cost_ch
        
        self.pointconv_list = nn.ModuleList()
        
        for _, ch_out in enumerate(channels):
            pointconv = PointConv(neighbors, last_channel + 3, ch_out, bn = True, use_leaky = True)
            self.pointconv_list.append(pointconv)
            last_channel = ch_out
        
        
        
        # self.pointconv = PointConv(9, last_channel + 3, mid_planes, bn = True, use_leaky = True)
        
        
        
        # 输出维度控制为64
        self.planes = planes 
        self.linear1 = nn.Linear(mid_planes, mid_planes, bias=False) # 由输入维度如512维度降到128中间维度
        self.bn1 = nn.BatchNorm1d(mid_planes)

        # self.linear2 = nn.Linear(mid_planes, planes, bias=False) # 由128中间维度降到64输出维度
        # self.bn2 = nn.BatchNorm1d(planes)

        self.transformer3 = PointTransformerLayer(mid_planes, mid_planes, share_planes, nsample)
        self.bn3 = nn.BatchNorm1d(mid_planes)

        self.linear4 = nn.Linear(mid_planes, mid_planes , bias=False)
        self.bn4 = nn.BatchNorm1d(mid_planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv = nn.Conv1d(mid_planes, planes, 1)

        self.fc = nn.Conv1d(planes, 3, 1)

    def forward(self, xyz, feats, cost_volume, flow = None):
        '''
        Input:
        xyz : B 3 N
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        '''
        # B 256+256=512 256
        new_points = torch.cat([feats, cost_volume], dim = 1)       
        # new_points = self.pointconv(xyz, new_points) 
        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv(xyz, new_points)
               
        xyz = xyz.transpose(1, 2) # [B, N ,3]
        points = new_points.transpose(1, 2)  # [B, N ,C=128]

        B, N, _ = points.shape

        points = self.linear1(points) # B N mid_planes = 128
        # 当input的维度为（N, C）时，BN将对C维归一化；当input的维度为(N, C, L) 时，归一化的维度同样为C维
        points = self.bn1(points.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        points = self.relu(points)

        # points = self.linear2(points) # B 8192 mid_planes = 64
        # # 当input的维度为（N, C）时，BN将对C维归一化；当input的维度为(N, C, L) 时，归一化的维度同样为C维
        # points = self.bn2(points.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        # points = self.relu(points)

        identity = points # B N 128

        points = self.relu(self.bn3(self.transformer3(xyz, points))) # B*N 128
        points = self.bn4(self.linear4(points))
        points = points.view(B,N,self.mid_planes)

        # [B, N ,C]
        points += identity
        points = self.relu(points) 

        new_points = points.transpose(1, 2)   # [B, C=128, N]
        
        new_points = self.conv(new_points)

        flow_local = self.fc(new_points).clamp(self.clamp[0], self.clamp[1]) 
        
        if flow is None:
            flow = flow_local
        else: 
            flow = flow_local + flow
        return new_points, flow
    





# My edit 
class new_pt_SceneFlowEstimatorResidual(nn.Module):
    def __init__(self, feat_ch, cost_ch, neighbors = 9,channels = [128, 128], mlp = [128, 64], mid_planes = 128, planes = 64, share_planes=8 , nsample = 16, clamp = [-200, 200], use_leaky = True):
    # def __init__(self, feat_ch, cost_ch, flow_ch = 3, channels = [128, 128], mlp = [128, 64], neighbors = 9, clamp = [-200, 200], use_leaky = True):
        super(new_pt_SceneFlowEstimatorResidual, self).__init__()
        self.use_leaky = use_leaky
        self.clamp = clamp
        self.mid_planes = mid_planes
        last_channel = feat_ch + cost_ch
        
        # self.pointconv_list = nn.ModuleList()
        
        # for _, ch_out in enumerate(channels):
        #     pointconv = PointConv(neighbors, last_channel + 3, ch_out, bn = True, use_leaky = True)
        #     self.pointconv_list.append(pointconv)
        #     last_channel = ch_out
        
        
        
        # self.pointconv = PointConv(9, last_channel + 3, mid_planes, bn = True, use_leaky = True)
        
        
        
        # 输出维度控制为64
        self.planes = planes 
        self.linear1 = nn.Linear(last_channel, mid_planes, bias=False) # 由输入维度如512维度降到128中间维度
        self.bn1 = nn.BatchNorm1d(mid_planes)

        # self.linear2 = nn.Linear(mid_planes, planes, bias=False) # 由128中间维度降到64输出维度
        # self.bn2 = nn.BatchNorm1d(planes)

        self.transformer3 = PointTransformerLayer(mid_planes, mid_planes, share_planes, nsample)
        self.bn3 = nn.BatchNorm1d(mid_planes)

        self.linear4 = nn.Linear(mid_planes, mid_planes , bias=False)
        self.bn4 = nn.BatchNorm1d(mid_planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv = nn.Conv1d(mid_planes, planes, 1)

        self.fc = nn.Conv1d(planes, 3, 1)

    def forward(self, xyz, feats, cost_volume, flow = None):
        '''
        Input:
        xyz : B 3 N
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        '''
        # B 256+256=512 256
        new_points = torch.cat([feats, cost_volume], dim = 1)       
        # new_points = self.pointconv(xyz, new_points) 
        # for _, pointconv in enumerate(self.pointconv_list):
        #     new_points = pointconv(xyz, new_points)
               
        xyz = xyz.transpose(1, 2) # [B, N ,3]
        points = new_points.transpose(1, 2)  # [B, N ,C=256+256]

        B, N, _ = points.shape

        points = self.linear1(points) # B N mid_planes = 128
        # 当input的维度为（N, C）时，BN将对C维归一化；当input的维度为(N, C, L) 时，归一化的维度同样为C维
        points = self.bn1(points.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        points = self.relu(points)

        # points = self.linear2(points) # B 8192 mid_planes = 64
        # # 当input的维度为（N, C）时，BN将对C维归一化；当input的维度为(N, C, L) 时，归一化的维度同样为C维
        # points = self.bn2(points.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        # points = self.relu(points)

        # identity = points # B N 128

        points = self.relu(self.bn3(self.transformer3(xyz, points))) # B*N 128
        points = self.bn4(self.linear4(points))
        points = points.view(B,N,self.mid_planes)

        # [B, N ,C]
        # points += identity
        points = self.relu(points) 

        new_points = points.transpose(1, 2)   # [B, C=128, N]
        
        new_points = self.conv(new_points)

        flow_local = self.fc(new_points).clamp(self.clamp[0], self.clamp[1]) 
        
        if flow is None:
            flow = flow_local
        else: 
            flow = flow_local + flow
        return new_points, flow