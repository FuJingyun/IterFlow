import torch
import torch.nn as nn
from .scatter import PointPillarsScatter, my_PointPillarsScatter
# import torch.nn.functional as F
from .bev_shift import shift_bev_grids, return_tensor_index

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



class Bi_cross_Attention(nn.Module):
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
        self.atten_fusion1 = torch.nn.MultiheadAttention(embed_dim=32, num_heads=self.head_num, dropout=0.1, batch_first=True)
        # STEP 2 : lidar to radar
        self.atten_fusion2 = torch.nn.MultiheadAttention(embed_dim=32, num_heads=self.head_num, dropout=0.1, batch_first=True)

        self.scatter = my_PointPillarsScatter(in_channels=feat_channels, output_shape=pseudo_image_dims)

    def forward(self, li_bev_feats , li_bev_coors, ra_bev_feats , ra_bev_coors) :

        li_pseudoimage_lst = []
        ra_pseudoimage_lst = []
       

        for id in range(len(li_bev_feats)):
            # [L, 32]
            # cur_li_bev_feat = li_bev_feats[id]
            cur_li_bev_feat = self.li_norm(li_bev_feats[id])

            # [L, 3]
            cur_li_bev_coor = li_bev_coors[id]

            # [R, 32]
            # cur_ra_bev_feat = ra_bev_feats[id]
            cur_ra_bev_feat = self.ra_norm(ra_bev_feats[id])

            # [R, 3]
            cur_ra_bev_coor = ra_bev_coors[id]


            cur_dev = cur_li_bev_feat.get_device()
            
            # STEP 1 : radar to lidar
            q_map, k_map, v_map = self.qkv1(cur_li_bev_feat, cur_ra_bev_feat)
            key_list = []
            value_list = []
            # [9, L, 2]
            shifted_index, shift_offset_ten = shift_bev_grids(cur_li_bev_coor, INDEX_SHIFT, self.pseudo_image_dims, cur_dev, v_map.dtype)
            
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
            out, _ = self.atten_fusion1(feat_query, key, value) # (L, 1, CHANNEL)
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
            shifted_index, shift_offset_ten = shift_bev_grids(cur_ra_bev_coor, INDEX_SHIFT, self.pseudo_image_dims, cur_dev, v_map.dtype)
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



class Dy_cross_Attention(nn.Module):
    def __init__(self,  pseudo_image_dims, 
                 feat_channels: int) -> None:
        super(Dy_cross_Attention, self).__init__()

        self.pseudo_image_dims = pseudo_image_dims
        self.feat_channels = feat_channels
        
        self.li_norm = LayerNorm(normalized_shape = feat_channels)
        self.ra_norm = LayerNorm(normalized_shape = feat_channels)
        # 
        self.qkv1 = qkv_encoder(feat_channels = feat_channels)
        self.qkv2 = qkv_encoder(feat_channels = feat_channels)

        self.position_encoder = nn.Linear(2, 32)

        self.head_num = 2
        # STEP 1 : radar to lidar
        self.atten_fusion1 = torch.nn.MultiheadAttention(embed_dim=32, num_heads=self.head_num, dropout=0.1, batch_first=True)
        # STEP 2 : lidar to radar
        self.atten_fusion2 = torch.nn.MultiheadAttention(embed_dim=32, num_heads=self.head_num, dropout=0.1, batch_first=True)

        self.scatter = my_PointPillarsScatter(in_channels=feat_channels, output_shape=pseudo_image_dims)
    
    def gen_dy_coor(self, li_coors, ra_dys, cur_device):
        # dy_li = []
        li_len = li_coors.shape[0]
        dy_li_mask = torch.zeros(li_len, dtype=torch.bool, device=cur_device)
        grid_size = torch.tensor([self.pseudo_image_dims[0]+1, self.pseudo_image_dims[1]+1], device=cur_device)
        shift_tensor = gen_index_shift(1, cur_device)

        for id in range(li_len):
            li_coor = li_coors[id]
            for shift in shift_tensor:
                shifted_coor = (li_coor + shift) % grid_size
                temp_x = shifted_coor[0]
                temp_y = shifted_coor[1]
                if (temp_x<self.pseudo_image_dims[0]) and (temp_y<self.pseudo_image_dims[1]):
                    if ra_dys[0,shifted_coor[0],shifted_coor[1]]:
                        dy_li_mask[id] = True
                        break          
        return dy_li_mask
    
    def old_gen_dy_coor(self, li_coors, ra_coors, ra_dys, cur_device):

        # dy_li = []
        li_len = li_coors.shape[0]
        ra_len = ra_coors.shape[0]
        dy_li_mask = torch.zeros(li_len, dtype=torch.bool, device=cur_device)
        grid_size = torch.tensor([self.pseudo_image_dims[0]+1, self.pseudo_image_dims[1]+1], device=cur_device)
        shift_tensor = gen_index_shift(2, cur_device)

        for id in range(li_len):
            li_coor = li_coors[id]
            for shift in shift_tensor:
                # shifted_coor = (li_coor + torch.tensor(shift, device=cur_device)) % grid_size
                shifted_coor = (li_coor + shift) % grid_size
                res = torch.all(ra_coors == shifted_coor, dim=1) # (R, 2) == (1,2) Res: R
                one = torch.ones(1, dtype=torch.bool, device=cur_device)
                res_with_one = torch.concat((res, one), dim=0) # (R+1)
                res_index = torch.nonzero(res_with_one)
                # if res_index is not None:
                #     if ra_dys[res_index[0]]:
                #         dy_li_mask[id] = True
                #         break
                if res_index[0]!= ra_len:
                    if ra_dys[res_index[0]]:
                        dy_li_mask[id] = True
                        break
           
        return dy_li_mask
    

    def new_gen_dy_coor(self, li_coors, ra_coors, ra_dys, cur_device):
        dy_ra_coor = ra_coors[ra_dys]
        li_len = li_coors.shape[0]
        grid_size = torch.tensor([self.pseudo_image_dims[0]+1, self.pseudo_image_dims[1]+1], device=cur_device)
        dy_li_mask = torch.zeros(li_len, dtype=torch.bool, device=cur_device)
        shift_tensor = gen_index_shift(2, cur_device)
        for id in range(dy_ra_coor.shape[0]):
            temp_dy_ra_coor = dy_ra_coor[id]
            for shift in shift_tensor:
                shifted_coor = (temp_dy_ra_coor + shift) % grid_size
                shifted_x = shifted_coor[0]
                shifted_y = shifted_coor[1]
                if (shifted_x<self.pseudo_image_dims[0]) and (shifted_y<self.pseudo_image_dims[1]):
                    res = torch.all(li_coors == shifted_coor, dim=1) # L
                    one = torch.ones(1, dtype=torch.bool, device=cur_device)
                    res_with_one = torch.concat((res, one), dim=0) # (L+1)
                    res_index = torch.nonzero(res_with_one)
                    if res_index[0]!= li_len:
                        dy_li_mask[res_index[0]] = True
     
        return dy_li_mask


    def forward(self, li_bev_feats , li_bev_coors, ra_bev_feats , ra_bev_coors, radar_bev_dy) :

        li_pseudoimage_lst = []
        ra_pseudoimage_lst = []
       

        for id in range(len(li_bev_feats)):
            # [L, 2]
            cur_li_bev_coor = li_bev_coors[id]
            # [L, 32]
            # cur_li_bev_feat = li_bev_feats[id]
            cur_li_bev_feat = self.li_norm(li_bev_feats[id])         

            # [R, 32]
            # cur_ra_bev_feat = ra_bev_feats[id]
            cur_ra_bev_feat = self.ra_norm(ra_bev_feats[id])
            # [R, 2]
            cur_ra_bev_coor = ra_bev_coors[id]
            # # [R, 1]
            # [1, H, W]
            cur_ra_dy = radar_bev_dy[id]

            cur_dev = cur_li_bev_feat.get_device()
            

            dy_li_mask = self.new_gen_dy_coor(cur_li_bev_coor, cur_ra_bev_coor, cur_ra_dy, cur_dev)

            li_dy_bev_num = torch.sum(dy_li_mask)
            # print("li_dy_bev_num")
            # print(li_dy_bev_num)

            if (li_dy_bev_num>0):
                dy_li_feat = cur_li_bev_feat[dy_li_mask]
                dy_li_coor = cur_li_bev_coor[dy_li_mask]

                # STEP 1 : radar to lidar
                q_map, k_map, v_map = self.qkv1(dy_li_feat, cur_ra_bev_feat)
                key_list = []
                value_list = []
                # [9, L, 2]
                shifted_index, shift_offset_ten = shift_bev_grids(dy_li_coor, INDEX_SHIFT, self.pseudo_image_dims, cur_dev, v_map.dtype)
                
                for i, each_shift_index in enumerate(shifted_index):
                    select_ind = return_tensor_index(value = each_shift_index, t=cur_ra_bev_coor) # [L]
                    # select_ind = torch.tensor(select_ind, device = cur_dev)
                    select_ind = select_ind.to(device = cur_dev)
                    condition = (select_ind >= 0).unsqueeze(1).expand_as(dy_li_feat) # (L, CHANNEL=32)

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
                out, _ = self.atten_fusion1(feat_query, key, value) # (L, 1, CHANNEL)
                out = out.squeeze(1) # (N, CHANNEL=32)

                # out_li_bev_feats = cur_li_bev_feat + out
                # [L,1]
                zero_voxel_channel0 = torch.zeros((dy_li_coor.shape[0],1), dtype=cur_li_bev_coor.dtype, device=cur_dev)
                cur_li_voxel_coor = torch.cat((zero_voxel_channel0, dy_li_coor), dim=1)

                pseudoimage = self.scatter(out, cur_li_voxel_coor)
                li_pseudoimage_lst.append(pseudoimage)
            else:
                # [B 32 H W]
                # torch.Size([1, 32, 512, 512])
                pseudoimage = torch.zeros((1,self.feat_channels,self.pseudo_image_dims[0],self.pseudo_image_dims[1]), dtype=cur_li_bev_feat.dtype, device=cur_dev)
                li_pseudoimage_lst.append(pseudoimage)


            # STEP 2 : lidar to radar
            q_map, k_map, v_map = self.qkv2(cur_ra_bev_feat, cur_li_bev_feat)
            key_list = []
            value_list = []
            # [9, R, 2]
            shifted_index, shift_offset_ten = shift_bev_grids(cur_ra_bev_coor, INDEX_SHIFT, self.pseudo_image_dims, cur_dev, v_map.dtype)
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
            out, _ = self.atten_fusion2(feat_query, key, value) # (R, 1, CHANNEL)
            out = out.squeeze(1)
            # out_ra_bev_feats = cur_ra_bev_feat + out

            # [R,1]
            zero_voxel_channel0 = torch.zeros((cur_ra_bev_coor.shape[0],1), dtype=cur_ra_bev_coor.dtype, device=cur_dev)
            cur_ra_voxel_coor = torch.cat((zero_voxel_channel0, cur_ra_bev_coor), dim=1)

            # torch.Size([1, 32, 512, 512])
            pseudoimage = self.scatter(out, cur_ra_voxel_coor)
            ra_pseudoimage_lst.append(pseudoimage)

                          
        # Concatenate the pseudoimages along the batch dimension
        return torch.cat(li_pseudoimage_lst, dim=0), torch.cat(ra_pseudoimage_lst, dim=0)



class Bi_Aug(nn.Module):
    def __init__(self,  pseudo_image_dims, 
                 feat_channels: int) -> None:
        super().__init__()

        self.pseudo_image_dims = pseudo_image_dims
        
        # self.li_norm = LayerNorm(normalized_shape = feat_channels)
        # self.ra_norm = LayerNorm(normalized_shape = feat_channels)
        # # 
        # self.qkv1 = qkv_encoder(feat_channels = feat_channels)
        # self.qkv2 = qkv_encoder(feat_channels = feat_channels)


        # self.head_num = 2
        # # STEP 1 : radar to lidar
        # self.atten_fusion1 = torch.nn.MultiheadAttention(embed_dim=32, num_heads=self.head_num, dropout=0.1, batch_first=True)
        # # STEP 2 : lidar to radar
        # self.atten_fusion2 = torch.nn.MultiheadAttention(embed_dim=32, num_heads=self.head_num, dropout=0.1, batch_first=True)

        self.scatter = PointPillarsScatter(in_channels=feat_channels, output_shape=pseudo_image_dims)

    def forward(self, li_bev_feats , li_bev_coors, ra_bev_feats , ra_bev_coors, pos_embedding) :

        li_pseudoimage_lst = []
        ra_pseudoimage_lst = []

        for id in range(len(li_bev_feats)):
            # [L, 32]
            cur_li_bev_feat = li_bev_feats[id]
            # cur_li_bev_feat = self.li_norm(li_bev_feats[id])
            # [L, 3]
            cur_li_bev_coor = li_bev_coors[id]
            # [R, 32]
            cur_ra_bev_feat = ra_bev_feats[id]
            # cur_ra_bev_feat = self.ra_norm(ra_bev_feats[id])
            # [R, 3]
            cur_ra_bev_coor = ra_bev_coors[id]
            cur_dev = cur_li_bev_feat.get_device()
            
            # STEP 1 : radar to lidar
            # q_map, k_map, v_map = self.qkv1(cur_li_bev_feat, cur_ra_bev_feat)
            # key_list = []
            value_list = []
            # [9, L, 2]
            shifted_index = shift_bev_grids(cur_li_bev_coor, INDEX_SHIFT, self.pseudo_image_dims, cur_dev)
            for i, each_shift_index in enumerate(shifted_index):
                select_ind = return_tensor_index(value = each_shift_index, t=cur_ra_bev_coor) # [L]
                # select_ind = torch.tensor(select_ind, device = cur_dev)
                select_ind = select_ind.to(device = cur_dev)
                condition = (select_ind >= 0).unsqueeze(1).expand_as(cur_li_bev_feat) # (L, CHANNEL=32)

                tmp_v = cur_ra_bev_feat[select_ind]  # (L, CHANNEL=32)
                # tmp_v = cur_ra_bev_feat[select_ind] + pos_embedding[i] # (L, CHANNEL=32)
                # tmp_k = cur_ra_bev_feat[select_ind]  # (L, CHANNEL=32)
                # tmp_v = v_map[select_ind] + pos_embedding[i] # (L, CHANNEL=32)
                # tmp_k = k_map[select_ind]  # (L, CHANNEL=32)
                tmp_v = tmp_v.masked_fill(~condition, 0) # fill zero if select_ind  == -1
                # tmp_k = tmp_k.masked_fill(~condition, 0) # fill zero if select_ind  == -1
                # value_list.append(tmp_v)
                # key_list.append(tmp_k)
                
            value = torch.stack(value_list) # [9, L , 32]
            # key = torch.stack(key_list) # [9, L , 32]
            value = value.permute(1, 0, 2) # (L, 9, CHANNEL=32)
            key = key.permute(1, 0, 2) # (L, 9, CHANNEL=32)

            # feat_query = cur_li_bev_feat.unsqueeze(1) # (L, 1, CHANNEL=32)
            feat_query = q_map.unsqueeze(1) # (L, 1, CHANNEL=32)
            out, _ = self.atten_fusion1(feat_query, key, value) # (L, 1, CHANNEL)
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
            shifted_index = shift_bev_grids(cur_ra_bev_coor, INDEX_SHIFT, self.pseudo_image_dims, cur_dev)
            for i, each_shift_index in enumerate(shifted_index):
                select_ind = return_tensor_index(value = each_shift_index, t=cur_li_bev_coor) # [R]
                # select_ind = torch.tensor(select_ind, device = cur_dev)
                select_ind = select_ind.to(device = cur_dev)
                condition = (select_ind >= 0).unsqueeze(1).expand_as(cur_ra_bev_feat) # (R, CHANNEL=32)

                # tmp_v = cur_li_bev_feat[select_ind] + pos_embedding[i] # (R, CHANNEL=32)
                # tmp_k = cur_li_bev_feat[select_ind]  # (R, CHANNEL=32)
                tmp_v = v_map[select_ind] + pos_embedding[i] # (R, CHANNEL=32)
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






class Self_Attention(nn.Module):
    def __init__(self,  pseudo_image_dims, 
                 feat_channels: int) -> None:
        super().__init__()

        self.pseudo_image_dims = pseudo_image_dims
        
        self.norm = LayerNorm(normalized_shape = feat_channels)

        self.qkv = qkv_encoder(feat_channels = feat_channels)

        self.position_encoder = nn.Linear(2, 32)

        self.head_num = 2
        self.atten_fusion = torch.nn.MultiheadAttention(embed_dim=32, num_heads=self.head_num, dropout=0.1, batch_first=True)

        self.scatter = my_PointPillarsScatter(in_channels=feat_channels, output_shape=pseudo_image_dims)

    def forward(self, bev_feats , bev_coors) :

        pseudoimage_lst = []
        
        for id in range(len(bev_feats)):
            # [L, 32]
            cur_bev_feat = self.norm(bev_feats[id])

            # [L, 3]
            cur_bev_coor = bev_coors[id]

            cur_dev = cur_bev_feat.get_device()
            
            q_map, k_map, v_map = self.qkv(cur_bev_feat, cur_bev_feat)
            key_list = []
            value_list = []
            # [9, L, 2]
            shifted_index, shift_offset_ten = shift_bev_grids(cur_bev_coor, INDEX_SHIFT, self.pseudo_image_dims, cur_dev, v_map.dtype)
            
            for i, each_shift_index in enumerate(shifted_index):
                select_ind = return_tensor_index(value = each_shift_index, t=cur_bev_coor) # [L]
                # select_ind = torch.tensor(select_ind, device = cur_dev)
                select_ind = select_ind.to(device = cur_dev)
                condition = (select_ind >= 0).unsqueeze(1).expand_as(cur_bev_feat) # (L, CHANNEL=32)

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
            out, _ = self.atten_fusion(feat_query, key, value) # (L, 1, CHANNEL)
            out = out.squeeze(1) # (N, CHANNEL=32)

            # out_li_bev_feats = cur_li_bev_feat + out
            # [L,1]
            zero_voxel_channel0 = torch.zeros((cur_bev_coor.shape[0],1), dtype=cur_bev_coor.dtype, device=cur_dev)
            cur_li_voxel_coor = torch.cat((zero_voxel_channel0, cur_bev_coor), dim=1)

            pseudoimage = self.scatter(out, cur_li_voxel_coor)
            pseudoimage_lst.append(pseudoimage)
                
        # Concatenate the pseudoimages along the batch dimension
        return torch.cat(pseudoimage_lst, dim=0)