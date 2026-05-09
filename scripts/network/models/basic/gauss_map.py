import torch
# import scipy.spatial as spt
import torch.nn as nn
from .scatter import PointPillarsScatter, my_PointPillarsScatter
import torch.nn.functional as F
from .bev_shift import shift_bev_grids, return_tensor_index


from torch import Tensor
from typing import Optional, Tuple, List
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
import math


INDEX_SHIFT = [ [0,0], [-1,0],[1,0], [0,1],[-1,1],[1,1],[0,-1],[-1,-1],[1,-1] ]

def gen_index_shift(size, cur_dev):
    start = -size
    end = size+1
    out_shift = []
    for i in range(start, end):
        for j in range(start, end):
            temp_shift = torch.zeros(2, dtype=torch.int, device=cur_dev)
            temp_shift[0] = i
            temp_shift[1] = j
            out_shift.append(temp_shift)
    return out_shift



class gauss_map(nn.Module):
    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range):
        super().__init__()

        self.pseudo_image_dims = pseudo_image_dims

    def neighbour_coors(self, bev_grids, shift_tensor, cur_device):
        """
        bev_coors : param voxel_grids: [N, 2], where N represents the number of non-sparse pillars
        shift_tensor: param shift_tensor: [9, 2]
        grid_size: param grid_size: [2]  
        cur_device : param current device
        return: shifted_voxel_grids: [N, 9, 2]
        """

        grid_size =  self.pseudo_image_dims
        shifted_bev_tensors = []
        # grid_size: [512,128] 
        # voxel_size:[0.4, 0.1, 6]
        # bs_grid_size = torch.tensor([1] + grid_size, device=cur_device)
        bs_grid_size = torch.tensor([grid_size[0]+1, grid_size[1]+1], device=cur_device)
        for shift in shift_tensor:
            shifted_bev_grids = (bev_grids + torch.tensor(shift, device=cur_device)) % bs_grid_size
            shifted_bev_tensors.append(shifted_bev_grids)
        return torch.stack(shifted_bev_tensors, dim=0).transpose(0,1)
    
    def nearest_dy(self, neighbour_coor, dy_coor, cur_device):
        # neighbour_coor:# [N, 9, 2]
        # dy_coor : [M ,2]
        num_pts = neighbour_coor.shape[0]
        num_neighbour = neighbour_coor.shape[1]
        dis_tensor = torch.zeros((num_pts, num_neighbour), device = cur_device)

        for i in range(num_pts):
            for j in range(num_neighbour):
                cur_coor = neighbour_coor[i,j,:]
                all_dis = torch.linalg.vector_norm((dy_coor.float() - cur_coor.float()), dim=-1)
                dis_tensor[i,j] = torch.min(all_dis)*0.01 # grid aize 0.1m

        return dis_tensor

                

    def forward(self, li_bev_coors, ra_bev_coors, ra_voxel_info_list) :

        lidar_gauss_score = []
        radar_gauss_score = []
       

        for id in range(len(ra_voxel_info_list)):
        # List of points and coordinates for each batch
        # result_dict = {
        #         "points": valid_batch_non_nan_points,
        #         "voxel_coords": valid_batch_voxel_coords,
        #         "point_idxes": valid_point_idxes,
        #         "point_offsets": point_offsets
        #     }
            # [L, 2]
            cur_li_bev_coor = li_bev_coors[id].contiguous()
            
            # [R, 2]
            cur_ra_bev_coor = ra_bev_coors[id]

            # [R_pts, 3]
            cur_ra_pts = ra_voxel_info_list[id]["points"]
            cur_ra_pts_coor = ra_voxel_info_list[id]["voxel_coords"]


            cur_dev = cur_li_bev_coor.get_device()

            ra_points_dy = torch.abs(cur_ra_pts[:,4]) > 0.1

            if torch.sum(ra_points_dy)>1 :
                dy_coors = cur_ra_pts_coor[ra_points_dy][:,1:]
                
                # # [R_dy_coor, 2]
                dy_coor = torch.unique(dy_coors, dim=0) # torch.Size([21, 2])

                # index_shift= gen_index_shift(2, cur_dev) # INDEX_SHIFT
                index_shift = INDEX_SHIFT

                li_neighbour_coor = self.neighbour_coors(bev_grids = cur_li_bev_coor, shift_tensor = index_shift, \
                                                        cur_device = cur_dev) # [L, 9, 2]
                
                ra_neighbour_coor = self.neighbour_coors(bev_grids = cur_ra_bev_coor, shift_tensor = index_shift, \
                                                        cur_device = cur_dev) # [R, 9, 2]

                li_dy_dis = self.nearest_dy(li_neighbour_coor, dy_coor, cur_dev)    
                ra_dy_dis = self.nearest_dy(ra_neighbour_coor, dy_coor, cur_dev) 

            else:
                li_dy_dis = torch.zeros((cur_li_bev_coor.shape[0], cur_li_bev_coor.shape[1]), device = cur_dev)
                ra_dy_dis = torch.zeros((cur_ra_bev_coor.shape[0], cur_ra_bev_coor.shape[1]), device = cur_dev)

            lidar_gauss_score.append(li_dy_dis)
            radar_gauss_score.append(ra_dy_dis)
                        
        # Concatenate the pseudoimages along the batch dimension
        return lidar_gauss_score, radar_gauss_score



# Input size B N C
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(LayerNorm, self).__init__()

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class qkv_encoder(nn.Module):
    def __init__(self, feat_channels):
        super(qkv_encoder, self).__init__()
        self.q_conv = nn.Conv1d(in_channels=feat_channels, out_channels = feat_channels, kernel_size = 1)
        self.k_conv = nn.Conv1d(in_channels=feat_channels, out_channels = feat_channels, kernel_size = 1)
        self.v_conv = nn.Conv1d(in_channels=feat_channels, out_channels = feat_channels, kernel_size = 1)

    def forward(self, x, y): # x as Q, y as K V
        q_map = x.permute(1,0) # [32, L]
        q_map = q_map.unsqueeze(0) #  [1, 32, L]
        q_map = self.q_conv(q_map) # [1, 32, L]

        y_t = y.permute(1,0)
        y_t = y_t.unsqueeze(0)
        k_map = self.k_conv(y_t)
        v_map = self.v_conv(y_t) # [1 C N]

        return (q_map.squeeze(0)).permute(1,0), (k_map.squeeze(0)).permute(1,0), (v_map.squeeze(0)).permute(1,0)

def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    return torch._C._nn.linear(input, weight, bias)


def in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:

    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            return linear(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (linear(q, w_q, b_q),) + linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)





def gauss_scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    sigma: Tensor,
    gauss: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:

    B, Nt, E = q.shape   # N, 1, 16
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn += attn_mask
    gauss_weight =  torch.exp(-(gauss)*sigma)
    
    # mu = gauss_weight.mean(-1, keepdim=True)
    # var = gauss_weight.var(-1, keepdim=True, unbiased=False)
    # gauss_weight = (gauss_weight - mu) / torch.sqrt(var + 1e-5)

    # max_gauss_weight,_ = torch.max(gauss_weight, dim=1, keepdim=True)
    # min_gauss_weight,_ = torch.min(gauss_weight, dim=1, keepdim=True)

    # gauss_weight = torch.true_divide((gauss_weight - min_gauss_weight),(max_gauss_weight-min_gauss_weight))

    gauss_weight = gauss_weight.unsqueeze(1)

    

    # if gauss_weight.shape[2] == attn.shape[2]:
    #     attn[0:attn.shape[0]:2,:,:] = torch.mul(attn[0:attn.shape[0]:2,:,:], gauss_weight)
    #     attn[1:attn.shape[0]:2,:,:] = torch.mul(attn[1:attn.shape[0]:2,:,:], gauss_weight)
    #     # attn[0:attn.shape[0]:2,:,:] = attn[0:attn.shape[0]:2,:,:] + gauss_weight
    #     # attn[1:attn.shape[0]:2,:,:] = attn[1:attn.shape[0]:2,:,:] + gauss_weight

    if gauss_weight.shape[2] == attn.shape[2]:
        part1 = attn[0:attn.shape[0]:2,:,:]
        part2 = attn[1:attn.shape[0]:2,:,:]
        # print("part1.shape")
        # print(part1.shape)
        # print("part2.shape")
        # print(part2.shape)
        # print("gauss_weight.shape")
        # print(gauss_weight.shape)
        if part1.shape[0] == gauss_weight.shape[0]:
            attn[0:attn.shape[0]:2,:,:] = torch.mul(part1, gauss_weight)
        if part2.shape[0] == gauss_weight.shape[0]:
            attn[1:attn.shape[0]:2,:,:] = torch.mul(part2, gauss_weight)


    # N 1 9
    attn = F.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output



def gauss_multi_head_attention(
    sigma: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    gauss: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Optional[Tensor],
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True
) -> Tuple[Tensor, Optional[Tensor]]:

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape # (1,L,32)
    src_len, _, _ = key.shape # # (9,L,32)
    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"

    assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    q, k, v = in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
   

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    attn_output = gauss_scaled_dot_product_attention(q, k, v, sigma, gauss, dropout_p)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    return attn_output







class gauss_MultiheadAttention(Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True, batch_first=True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(gauss_MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim

        self._qkv_same_embed_dim = True

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        self.register_parameter('q_proj_weight', None)
        self.register_parameter('k_proj_weight', None)
        self.register_parameter('v_proj_weight', None)


        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
       
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        # self.sigma = Parameter(torch.rand(10), requires_grad=True)
        # self.sigma = Parameter(torch.empty(1, **factory_kwargs))
        self.sigma = 10

        self._reset_parameters()

        

    def _reset_parameters(self):

        xavier_uniform_(self.in_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

        # constant_(self.sigma, 10)


    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(gauss_MultiheadAttention, self).__setstate__(state)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, gauss:Tensor) -> Tuple[Tensor, Optional[Tensor]]:
       
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        attn_output = gauss_multi_head_attention(self.sigma,
            query, key, value, gauss, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training)
        
        if self.batch_first:
            return attn_output.transpose(1, 0)
        else:
            return attn_output




class gauss_attention(nn.Module):
    def __init__(self,  pseudo_image_dims, 
                 feat_channels: int) -> None:
        super().__init__()

        self.pseudo_image_dims = pseudo_image_dims
        
        self.li_norm = LayerNorm(normalized_shape = feat_channels)
        self.ra_norm = LayerNorm(normalized_shape = feat_channels)
        # 
        self.qkv1 = qkv_encoder(feat_channels = feat_channels)
        self.qkv2 = qkv_encoder(feat_channels = feat_channels)

        self.position_encoder = nn.Linear(2, 32)

        self.head_num = 2
        # STEP 1 : radar to lidar
        self.atten_fusion1 = gauss_MultiheadAttention(embed_dim=32, num_heads=self.head_num, dropout=0.1, batch_first=True)
        # STEP 2 : lidar to radar
        self.atten_fusion2 = gauss_MultiheadAttention(embed_dim=32, num_heads=self.head_num, dropout=0.1, batch_first=True)

        self.scatter = my_PointPillarsScatter(in_channels=feat_channels, output_shape=pseudo_image_dims)

    def forward(self, li_bev_feats , li_bev_coors, ra_bev_feats , ra_bev_coors, li_gauss, ra_gauss) :

        li_pseudoimage_lst = []
        ra_pseudoimage_lst = []
       

        for id in range(len(li_bev_feats)):
            # [L, 32]
            # cur_li_bev_feat = li_bev_feats[id]
            cur_li_bev_feat = self.li_norm(li_bev_feats[id])

            # [L, 3]
            cur_li_bev_coor = li_bev_coors[id]
            # [L, 9]
            cur_li_gauss = li_gauss[id]

            # [R, 32]
            # cur_ra_bev_feat = ra_bev_feats[id]
            cur_ra_bev_feat = self.ra_norm(ra_bev_feats[id])

            # [R, 3]
            cur_ra_bev_coor = ra_bev_coors[id]
            # [R, 9]
            cur_ra_gauss = ra_gauss[id]

            cur_dev = cur_li_bev_feat.get_device()


            
            # STEP 1 : radar to lidar
            q_map, k_map, v_map = self.qkv1(cur_li_bev_feat, cur_ra_bev_feat)
            key_list = []
            value_list = []
            # [9, L, 2]
            index_shift = INDEX_SHIFT
            # index_shift = gen_index_shift(2, cur_dev) # INDEX_SHIFT
            shifted_index, shift_offset_ten = shift_bev_grids(cur_li_bev_coor, index_shift, self.pseudo_image_dims, cur_dev, v_map.dtype)
            
            for i, each_shift_index in enumerate(shifted_index):
                select_ind = return_tensor_index(value = each_shift_index, t=cur_ra_bev_coor) # [L]
                # select_ind = torch.tensor(select_ind, device = cur_dev)
                select_ind = select_ind.to(device = cur_dev)
                condition = (select_ind >= 0).unsqueeze(1).expand_as(cur_li_bev_feat) # (L, CHANNEL=32)

                # tmp_v = cur_ra_bev_feat[select_ind] + pos_embedding[i] # (L, CHANNEL=32)
                # tmp_k = cur_ra_bev_feat[select_ind]  # (L, CHANNEL=32)
                tmp_v = v_map[select_ind] + self.position_encoder(shift_offset_ten[i]) # (L, CHANNEL=32)
                tmp_k = k_map[select_ind]  # (L, CHANNEL=32)
                tmp_v = tmp_v.masked_fill(~condition, 0) # fill zero if select_ind  == -1
                tmp_k = tmp_k.masked_fill(~condition, 0) # fill zero if select_ind  == -1
                value_list.append(tmp_v)
                key_list.append(tmp_k)
                
            value = torch.stack(value_list) # [9, L , 32]
            key = torch.stack(key_list) # [9, L , 32]
            value = value.permute(1, 0, 2) # (L, 9, CHANNEL=32)
            key = key.permute(1, 0, 2) # (L, 9, CHANNEL=32)

            # feat_query = cur_li_bev_feat.unsqueeze(1) # (L, 1, CHANNEL=32)
            feat_query = q_map.unsqueeze(1) # (L, 1, CHANNEL=32)
            out = self.atten_fusion1(feat_query, key, value, cur_li_gauss) # (L, 1, CHANNEL)
            out = out.squeeze(1) # (N, CHANNEL=32)

            # out_li_bev_feats = cur_li_bev_feat + out
            # [L,1]
            zero_voxel_channel0 = torch.zeros((cur_li_bev_coor.shape[0],1), dtype=cur_li_bev_coor.dtype, device=cur_dev)
            cur_li_voxel_coor = torch.cat((zero_voxel_channel0, cur_li_bev_coor), dim=1)

            pseudoimage = self.scatter(out, cur_li_voxel_coor)
            li_pseudoimage_lst.append(pseudoimage)


            # STEP 2 : lidar to radar
            q_map, k_map, v_map = self.qkv2(cur_ra_bev_feat, cur_li_bev_feat)
            key_list = []
            value_list = []
            # [9, R, 2]
            index_shift = INDEX_SHIFT
            # index_shift = gen_index_shift(2, cur_dev) # INDEX_SHIFT
            shifted_index, shift_offset_ten = shift_bev_grids(cur_ra_bev_coor, index_shift, self.pseudo_image_dims, cur_dev, v_map.dtype)
            for i, each_shift_index in enumerate(shifted_index):
                select_ind = return_tensor_index(value = each_shift_index, t=cur_li_bev_coor) # [R]
                # select_ind = torch.tensor(select_ind, device = cur_dev)
                select_ind = select_ind.to(device = cur_dev)
                condition = (select_ind >= 0).unsqueeze(1).expand_as(cur_ra_bev_feat) # (R, CHANNEL=32)

                # tmp_v = cur_li_bev_feat[select_ind] + pos_embedding[i] # (R, CHANNEL=32)
                # tmp_k = cur_li_bev_feat[select_ind]  # (R, CHANNEL=32)
                tmp_v = v_map[select_ind] + self.position_encoder(shift_offset_ten[i]) # (R, CHANNEL=32)
                tmp_k = k_map[select_ind]  # (R, CHANNEL=32)
                tmp_v = tmp_v.masked_fill(~condition, 0) # fill zero if select_ind  == -1
                tmp_k = tmp_k.masked_fill(~condition, 0) # fill zero if select_ind  == -1
                value_list.append(tmp_v)
                key_list.append(tmp_k)

            value = torch.stack(value_list) # [9, R, 32]
            key = torch.stack(key_list) # [9, R, 32]
            value = value.permute(1, 0, 2) # (R, 9, CHANNEL=32)
            key = key.permute(1, 0, 2) # (R , 9, CHANNEL=32)

            # feat_query = cur_ra_bev_feat.unsqueeze(1) # (R, 1, CHANNEL=32)
            feat_query = q_map.unsqueeze(1) # (R, 1, CHANNEL=32)
            out = self.atten_fusion2(feat_query, key, value, cur_ra_gauss) # (R, 1, CHANNEL)
            out = out.squeeze(1)
            # out_ra_bev_feats = cur_ra_bev_feat + out

            # [R,1]
            zero_voxel_channel0 = torch.zeros((cur_ra_bev_coor.shape[0],1), dtype=cur_ra_bev_coor.dtype, device=cur_dev)
            cur_ra_voxel_coor = torch.cat((zero_voxel_channel0, cur_ra_bev_coor), dim=1)

            pseudoimage = self.scatter(out, cur_ra_voxel_coor)
            ra_pseudoimage_lst.append(pseudoimage)

                          
        # Concatenate the pseudoimages along the batch dimension
        return torch.cat(li_pseudoimage_lst, dim=0), torch.cat(ra_pseudoimage_lst, dim=0)
