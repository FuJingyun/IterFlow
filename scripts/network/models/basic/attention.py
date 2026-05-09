import torch
import torch.nn as nn
from .scatter import PointPillarsScatter
# import torch.nn.functional as F
from .bev_shift import shift_bev_grids, return_tensor_index

INDEX_SHIFT = [ [0,0], [-1,0],[1,0], [0,1],[-1,1],[1,1],[0,-1],[-1,-1],[1,-1] ]

class pos_Bi_Attention(nn.Module):
    def __init__(self,  pseudo_image_dims, 
                 feat_channels: int) -> None:
        super().__init__()

        self.pseudo_image_dims = pseudo_image_dims
        self.head_num = 2
        # STEP 1 : radar to lidar
        self.atten_fusion1 = torch.nn.MultiheadAttention(embed_dim=32, num_heads=self.head_num, dropout=0.1, batch_first=True)
        # STEP 2 : lidar to radar
        self.atten_fusion2 = torch.nn.MultiheadAttention(embed_dim=32, num_heads=self.head_num, dropout=0.1, batch_first=True)

        self.scatter = PointPillarsScatter(in_channels=feat_channels, output_shape=pseudo_image_dims)

    def forward(self, li_bev_feats , li_bev_coors, ra_bev_feats , ra_bev_coors, pos_embedding) :

        li_pseudoimage_lst = []
        ra_pseudoimage_lst = []

        for id in range(len(li_bev_feats)):
            # [L, 32]
            cur_li_bev_feat = li_bev_feats[id]
            # [L, 3]
            cur_li_bev_coor = li_bev_coors[id]
            # [R, 32]
            cur_ra_bev_feat = ra_bev_feats[id]
            # [R, 3]
            cur_ra_bev_coor = ra_bev_coors[id]
            cur_dev = cur_li_bev_feat.get_device()
            
            # STEP 1 : radar to lidar
            key_list = []
            value_list = []
            # [9, L, 2]
            shifted_index = shift_bev_grids(cur_li_bev_coor, INDEX_SHIFT, self.pseudo_image_dims, cur_dev)
            for i, each_shift_index in enumerate(shifted_index):
                select_ind = return_tensor_index(value = each_shift_index, t=cur_ra_bev_coor) # [L]
                # select_ind = torch.tensor(select_ind, device = cur_dev)
                select_ind = select_ind.to(device = cur_dev)
                condition = (select_ind >= 0).unsqueeze(1).expand_as(cur_li_bev_feat) # (L, CHANNEL=32)

                tmp_v = cur_ra_bev_feat[select_ind] + pos_embedding[i] # (L, CHANNEL=32)
                tmp_k = cur_ra_bev_feat[select_ind]  # (L, CHANNEL=32)
                tmp_v = tmp_v.masked_fill(~condition, 0) # fill zero if select_ind  == -1
                tmp_k = tmp_k.masked_fill(~condition, 0) # fill zero if select_ind  == -1
                value_list.append(tmp_v)
                key_list.append(tmp_k)
                
            value = torch.stack(value_list) # [9, L , 32]
            key = torch.stack(key_list) # [9, L , 32]
            value = value.permute(1, 0, 2) # (L, 9, CHANNEL=32)
            key = key.permute(1, 0, 2) # (L, 9, CHANNEL=32)

            feat_query = cur_li_bev_feat.unsqueeze(1) # (L, 1, CHANNEL=32)
            out, _ = self.atten_fusion1(feat_query, key, value) # (L, 1, CHANNEL)
            out = out.squeeze(1) # (N, CHANNEL=32)

            # out_li_bev_feats = cur_li_bev_feat + out
            # [L,1]
            zero_voxel_channel0 = torch.zeros((cur_li_bev_coor.shape[0],1), dtype=cur_li_bev_coor.dtype, device=cur_dev)
            cur_li_voxel_coor = torch.cat((zero_voxel_channel0, cur_li_bev_coor), dim=1)

            pseudoimage = self.scatter(out, cur_li_voxel_coor)
            li_pseudoimage_lst.append(pseudoimage)


            # STEP 2 : lidar to radar
            key_list = []
            value_list = []
            # [9, R, 2]
            shifted_index = shift_bev_grids(cur_ra_bev_coor, INDEX_SHIFT, self.pseudo_image_dims, cur_dev)
            for i, each_shift_index in enumerate(shifted_index):
                select_ind = return_tensor_index(value = each_shift_index, t=cur_li_bev_coor) # [R]
                # select_ind = torch.tensor(select_ind, device = cur_dev)
                select_ind = select_ind.to(device = cur_dev)
                condition = (select_ind >= 0).unsqueeze(1).expand_as(cur_ra_bev_feat) # (R, CHANNEL=32)

                tmp_v = cur_li_bev_feat[select_ind] + pos_embedding[i] # (R, CHANNEL=32)
                tmp_k = cur_li_bev_feat[select_ind]  # (R, CHANNEL=32)
                tmp_v = tmp_v.masked_fill(~condition, 0) # fill zero if select_ind  == -1
                tmp_k = tmp_k.masked_fill(~condition, 0) # fill zero if select_ind  == -1
                value_list.append(tmp_v)
                key_list.append(tmp_k)

            value = torch.stack(value_list)
            key = torch.stack(key_list) # [9, R, 32]
            value = value.permute(1, 0, 2) # (R, 9, CHANNEL=32)
            key = key.permute(1, 0, 2) # (R , 9, CHANNEL=32)

            feat_query = cur_ra_bev_feat.unsqueeze(1) # (R, 1, CHANNEL=32)
            out, _ = self.atten_fusion2(feat_query, key, value) # (R, 1, CHANNEL)
            out = out.squeeze(1)
            # out_ra_bev_feats = cur_ra_bev_feat + out

            # [R,1]
            zero_voxel_channel0 = torch.zeros((cur_ra_bev_coor.shape[0],1), dtype=cur_ra_bev_coor.dtype, device=cur_dev)
            cur_ra_voxel_coor = torch.cat((zero_voxel_channel0, cur_ra_bev_coor), dim=1)

            pseudoimage = self.scatter(out, cur_ra_voxel_coor)
            ra_pseudoimage_lst.append(pseudoimage)

                          
        # Concatenate the pseudoimages along the batch dimension
        return torch.cat(li_pseudoimage_lst, dim=0), torch.cat(ra_pseudoimage_lst, dim=0)

class Bi_Attention(nn.Module):
    def __init__(self,  pseudo_image_dims, 
                 feat_channels: int) -> None:
        super().__init__()

        self.pseudo_image_dims = pseudo_image_dims
        self.head_num = 2
        # STEP 1 : radar to lidar
        self.atten_fusion1 = torch.nn.MultiheadAttention(embed_dim=32, num_heads=self.head_num, dropout=0.1, batch_first=True)
        # STEP 2 : lidar to radar
        self.atten_fusion2 = torch.nn.MultiheadAttention(embed_dim=32, num_heads=self.head_num, dropout=0.1, batch_first=True)

        self.scatter = PointPillarsScatter(in_channels=feat_channels, output_shape=pseudo_image_dims)

    def forward(self, li_bev_feats , li_bev_coors, ra_bev_feats , ra_bev_coors) :

        li_pseudoimage_lst = []
        ra_pseudoimage_lst = []

        for id in range(len(li_bev_feats)):
            # [L, 32]
            cur_li_bev_feat = li_bev_feats[id]
            # [L, 3]
            cur_li_bev_coor = li_bev_coors[id]
            # [R, 32]
            cur_ra_bev_feat = ra_bev_feats[id]
            # [R, 3]
            cur_ra_bev_coor = ra_bev_coors[id]
            cur_dev = cur_li_bev_feat.get_device()
            
            # STEP 1 : radar to lidar
            key_value_list = []
            # [9, L, 2]
            shifted_index = shift_bev_grids(cur_li_bev_coor, INDEX_SHIFT, self.pseudo_image_dims, cur_dev)
            for _, each_shift_index in enumerate(shifted_index):
                select_ind = return_tensor_index(value = each_shift_index, t=cur_ra_bev_coor) # [L]
                # select_ind = torch.tensor(select_ind, device = cur_dev)
                select_ind = select_ind.to(device = cur_dev)
                condition = (select_ind >= 0).unsqueeze(1).expand_as(cur_li_bev_feat) # (L, CHANNEL=32)

                tmp = cur_ra_bev_feat[select_ind] # (L, CHANNEL=32)
                tmp= tmp.masked_fill(~condition, 0) # fill zero if select_ind  == -1
                key_value_list.append(tmp)
            key = torch.stack(key_value_list) # [9, L , 32]
            key = key.permute(1, 0, 2) # (L, 9, CHANNEL=32)

            feat_query = cur_li_bev_feat.unsqueeze(1) # (L, 1, CHANNEL=32)
            out, _ = self.atten_fusion1(feat_query, key, key) # (L, 1, CHANNEL)
            out = out.squeeze(1)
            out_li_bev_feats = cur_li_bev_feat + out
            # [L,1]
            zero_voxel_channel0 = torch.zeros((cur_li_bev_coor.shape[0],1), dtype=cur_li_bev_coor.dtype, device=cur_dev)
            cur_li_voxel_coor = torch.cat((zero_voxel_channel0, cur_li_bev_coor), dim=1)

            pseudoimage = self.scatter(out_li_bev_feats, cur_li_voxel_coor)
            li_pseudoimage_lst.append(pseudoimage)


            # STEP 2 : lidar to radar
            key_value_list = []
            # [9, R, 2]
            shifted_index = shift_bev_grids(cur_ra_bev_coor, INDEX_SHIFT, self.pseudo_image_dims, cur_dev)
            for _, each_shift_index in enumerate(shifted_index):
                select_ind = return_tensor_index(value = each_shift_index, t=cur_li_bev_coor) # [R]
                # select_ind = torch.tensor(select_ind, device = cur_dev)
                select_ind = select_ind.to(device = cur_dev)
                condition = (select_ind >= 0).unsqueeze(1).expand_as(cur_ra_bev_feat) # (R, CHANNEL=32)

                tmp = cur_li_bev_feat[select_ind] # (R, CHANNEL=32)
                tmp= tmp.masked_fill(~condition, 0) # fill zero if select_ind  == -1
                key_value_list.append(tmp)
            key = torch.stack(key_value_list) # [9, R, 32]
            key = key.permute(1, 0, 2) # (R , 9, CHANNEL=32)

            feat_query = cur_ra_bev_feat.unsqueeze(1) # (R, 1, CHANNEL=32)
            out, _ = self.atten_fusion2(feat_query, key, key) # (R, 1, CHANNEL)
            out = out.squeeze(1)
            out_ra_bev_feats = cur_ra_bev_feat + out

            # [R,1]
            zero_voxel_channel0 = torch.zeros((cur_ra_bev_coor.shape[0],1), dtype=cur_ra_bev_coor.dtype, device=cur_dev)
            cur_ra_voxel_coor = torch.cat((zero_voxel_channel0, cur_ra_bev_coor), dim=1)

            pseudoimage = self.scatter(out_ra_bev_feats, cur_ra_voxel_coor)
            ra_pseudoimage_lst.append(pseudoimage)

                          
        # Concatenate the pseudoimages along the batch dimension
        return torch.cat(li_pseudoimage_lst, dim=0), torch.cat(ra_pseudoimage_lst, dim=0)

class Pillar_Self_Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.q = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : feature maps from feature extractor. (N, C, H, W)
        outputs :
            feature maps weighted by attention along spatial dimensions
        """
        N, C, H, W = x.shape
        assert C == self.in_channels

        query = self.q(x).view(N, -1, H*W).permute(0, 2, 1)  # (N, H*W, C')
        key = self.k(x).view(N, -1, H*W)  # (N, C', H*W)

        # caluculate correlation
        attention = torch.bmm(query, key)    # (N, H*W, H*W)
        # spatial normalize
        attention = self.softmax(attention)

        value = self.v(x).view(N, -1, H*W)    # (N, C, H*W)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(N, C, H, W)
        out = self.gamma*out + x
        return out



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        # self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.conv1 = nn.Sequential(
                    nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
                    nn.BatchNorm1d(1, eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio = 8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)