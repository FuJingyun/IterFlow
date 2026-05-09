import torch
import torch.nn as nn

from .make_voxels import  DynamicVoxelizer  # HardVoxelizer,
from .process_voxels import DynamicPillarFeatureNet, Fusion_DynamicPillarFeatureNet # DynamicPillarFeatureNet # PillarFeatureNet,  
from .scatter import PointPillarsScatter, my_PointPillarsScatter
from .bev_shift import shift_bev_grids, return_tensor_index
# import torch_scatter


INDEX_SHIFT = [ [0,0], [-1,0],[1,0], [0,1],[-1,1],[1,1],[0,-1],[-1,-1],[1,-1] ]

class pos_radar_DynamicEmbedder(nn.Module):
    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 feat_channels: int, mix = True, is_att = True) -> None:
        super().__init__()
        self.mix = mix
        self.is_att = is_att
        self.pseudo_image_dims = pseudo_image_dims
        
        self.voxelizer = DynamicVoxelizer(voxel_size=voxel_size,
                                          point_cloud_range=point_cloud_range)
        self.feature_net = Fusion_DynamicPillarFeatureNet(
            in_channels=6,
            feat_channels=(feat_channels,),  #  int(0.5*feat_channels), feat_channels,  # feat_channels,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            mode='avg',
            mix = self.mix) # 'avg'
        
        if self.is_att:
            self.head_num = 2
            self.atten_fusion = torch.nn.MultiheadAttention(embed_dim=32, num_heads=self.head_num, dropout=0.1, batch_first=True)

        self.scatter = PointPillarsScatter(in_channels=feat_channels, output_shape=pseudo_image_dims)

    def forward(self, points: torch.Tensor , pos_embedding: torch.Tensor) -> torch.Tensor:

        # List of points and coordinates for each batch
        # result_dict = {
        #         "points": valid_batch_non_nan_points,
        #         "voxel_coords": valid_batch_voxel_coords,
        #         "point_idxes": valid_point_idxes,
        #         "point_offsets": point_offsets
        #     }

        voxel_info_list = self.voxelizer(points)

        pseudoimage_lst = []
        pt_fea_lst = []
        out_bev_feats_lst = []
        out_bev_coors_lst = []

        for voxel_info_dict in voxel_info_list:
            points = voxel_info_dict['points']
            coordinates = voxel_info_dict['voxel_coords']
            # self.feature_net = DynamicPillarFeatureNet
           
            voxel_feats, voxel_coors, point_feats = self.feature_net(points, coordinates)
            # voxel_feats : torch.Size([NUM_non_empty_voxel, 32])
            # voxel_coors : torch.Size([NUM_non_empty_voxel, 3])   Example [ 0, 464,  99 ]
            # point_feats : torch.Size([ N_points , 32])
            pt_fea_lst.append(point_feats)

            if self.is_att:
                key_value_list = []
                cur_dev = voxel_coors.get_device()
                # [N, 2]
                bev_coors = voxel_coors[:,1:]
                # [N, 32]
                bev_feats = voxel_feats
                # [9, N, 2]
                shifted_index = shift_bev_grids(bev_coors, INDEX_SHIFT, self.pseudo_image_dims, cur_dev)
                # VERSION 2
                # ------------------------------------------------------------------------------------------------------
                for i, each_shift_index in enumerate(shifted_index):
                    select_ind = return_tensor_index(value = each_shift_index, t=bev_coors) # [N]
                    select_ind = torch.tensor(select_ind, device = cur_dev)
                    condition = (select_ind >= 0).unsqueeze(1).expand_as(bev_feats) # (N, CHANNEL=32)

                    # 相对位置编码
                    # cur_position = torch.tensor(INDEX_SHIFT[i], device = cur_dev)
                    # position_embedding = self.position_encoder(cur_position)
                    tmp = bev_feats[select_ind] + pos_embedding[i] # (N, CHANNEL=32)
                    tmp= tmp.masked_fill(~condition, 0) # fill zero if select_ind  == -1
                    key_value_list.append(tmp)
                key = torch.stack(key_value_list) # [9, N , 32]
                key = key.permute(1, 0, 2) # (N, 9, CHANNEL=32)

                feat_query = bev_feats.unsqueeze(1) # (N, 1, CHANNEL=32)
                out, _ = self.atten_fusion(feat_query, key, key) # (N, 1, CHANNEL)
                out = out.squeeze(1) # (N, CHANNEL=32)

                concat_new_voxel_feats = torch.cat([voxel_feats, out], dim=1) # (N, 64)

                # [64, H, W]
                pseudoimage = self.scatter(concat_new_voxel_feats, voxel_coors)
                pseudoimage_lst.append(pseudoimage)

                out_bev_feats_lst.append(bev_feats)
                out_bev_coors_lst.append(bev_coors)
                # ------------------------------------------------------------------------------------------------------
               
            else: 
                # [N, 32]
                bev_feats = voxel_feats
                # [N, 2]
                bev_coors = voxel_coors[:,1:]
                pseudoimage = self.scatter(voxel_feats, voxel_coors)
                # [32, H, W]
                pseudoimage_lst.append(pseudoimage)

                out_bev_feats_lst.append(bev_feats)
                out_bev_coors_lst.append(bev_coors)
            
        # Concatenate the pseudoimages along the batch dimension
        return torch.cat(pseudoimage_lst, dim=0), voxel_info_list, pt_fea_lst, out_bev_feats_lst, out_bev_coors_lst
    

class pos_lidar_DynamicEmbedder(nn.Module):
    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 feat_channels: int, mix = True, is_att = True) -> None:
        super().__init__()
        self.mix = mix
        self.is_att = is_att
        self.pseudo_image_dims = pseudo_image_dims
        
        self.voxelizer = DynamicVoxelizer(voxel_size=voxel_size,
                                          point_cloud_range=point_cloud_range)
        self.feature_net = Fusion_DynamicPillarFeatureNet(
            in_channels=3,
            feat_channels=(feat_channels,),  #  int(0.5*feat_channels), feat_channels,  # feat_channels,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            mode='avg',
            mix = self.mix) # 'avg'
        
        if self.is_att:
            self.head_num = 2
            self.atten_fusion = torch.nn.MultiheadAttention(embed_dim=32, num_heads=self.head_num, dropout=0.1, batch_first=True)

        self.scatter = PointPillarsScatter(in_channels=feat_channels, output_shape=pseudo_image_dims)

    def forward(self, points: torch.Tensor , pos_embedding: torch.Tensor) -> torch.Tensor:

        # List of points and coordinates for each batch
        # result_dict = {
        #         "points": valid_batch_non_nan_points,
        #         "voxel_coords": valid_batch_voxel_coords,
        #         "point_idxes": valid_point_idxes,
        #         "point_offsets": point_offsets
        #     }

        voxel_info_list = self.voxelizer(points)

        pseudoimage_lst = []
        pt_fea_lst = []
        out_bev_feats_lst = []
        out_bev_coors_lst = []

        for voxel_info_dict in voxel_info_list:
            points = voxel_info_dict['points']
            coordinates = voxel_info_dict['voxel_coords']
            # self.feature_net = DynamicPillarFeatureNet
           
            voxel_feats, voxel_coors, point_feats = self.feature_net(points, coordinates)
            # voxel_feats : torch.Size([NUM_non_empty_voxel, 32])
            # voxel_coors : torch.Size([NUM_non_empty_voxel, 3])   Example [ 0, 464,  99 ]
            # point_feats : torch.Size([ N_points , 32])
            pt_fea_lst.append(point_feats)

            if self.is_att:
                key_value_list = []
                cur_dev = voxel_coors.get_device()
                # [N, 2]
                bev_coors = voxel_coors[:,1:]
                # [N, 32]
                bev_feats = voxel_feats
                # [9, N, 2]
                shifted_index = shift_bev_grids(bev_coors, INDEX_SHIFT, self.pseudo_image_dims, cur_dev)
                # VERSION 2
                # ------------------------------------------------------------------------------------------------------
                for i, each_shift_index in enumerate(shifted_index):
                    select_ind = return_tensor_index(value = each_shift_index, t=bev_coors) # [N]
                    select_ind = torch.tensor(select_ind, device = cur_dev)
                    condition = (select_ind >= 0).unsqueeze(1).expand_as(bev_feats) # (N, CHANNEL=32)

                    # 相对位置编码
                    # cur_position = torch.tensor(INDEX_SHIFT[i], device = cur_dev)
                    # position_embedding = self.position_encoder(cur_position)
                    tmp = bev_feats[select_ind] + pos_embedding[i] # (N, CHANNEL=32)
                    tmp= tmp.masked_fill(~condition, 0) # fill zero if select_ind  == -1
                    key_value_list.append(tmp)
                key = torch.stack(key_value_list) # [9, N , 32]
                key = key.permute(1, 0, 2) # (N, 9, CHANNEL=32)

                feat_query = bev_feats.unsqueeze(1) # (N, 1, CHANNEL=32)
                out, _ = self.atten_fusion(feat_query, key, key) # (N, 1, CHANNEL)
                out = out.squeeze(1) # (N, CHANNEL=32)

                concat_new_voxel_feats = torch.cat([voxel_feats, out], dim=1) # (N, 64)

                # [64, H, W]
                pseudoimage = self.scatter(concat_new_voxel_feats, voxel_coors)
                pseudoimage_lst.append(pseudoimage)

                out_bev_feats_lst.append(bev_feats)
                out_bev_coors_lst.append(bev_coors)
                # ------------------------------------------------------------------------------------------------------
               
            else: 
                # [N, 32]
                bev_feats = voxel_feats
                # [N, 2]
                bev_coors = voxel_coors[:,1:]
                pseudoimage = self.scatter(voxel_feats, voxel_coors)
                # [32, H, W]
                pseudoimage_lst.append(pseudoimage)

                out_bev_feats_lst.append(bev_feats)
                out_bev_coors_lst.append(bev_coors)
            
        # Concatenate the pseudoimages along the batch dimension
        return torch.cat(pseudoimage_lst, dim=0), voxel_info_list, pt_fea_lst, out_bev_feats_lst, out_bev_coors_lst

class radar_DynamicEmbedder(nn.Module):
    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 feat_channels: int, mix = False, is_att = False) -> None:
        super().__init__()
        self.mix = mix
        self.is_att = is_att
        self.pseudo_image_dims = pseudo_image_dims
        
        self.voxelizer = DynamicVoxelizer(voxel_size=voxel_size,
                                          point_cloud_range=point_cloud_range)
        self.feature_net = Fusion_DynamicPillarFeatureNet(
            in_channels=5,  # differ 6
            feat_channels=(feat_channels,),  #  int(0.5*feat_channels), feat_channels,  # feat_channels,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            mode='avg', 
            mix = self.mix) # 'avg'
        
        self.head_num = 2
        if self.is_att:
            self.atten_fusion = torch.nn.MultiheadAttention(embed_dim=32, num_heads=self.head_num, dropout=0.1, batch_first=True)

        self.scatter = my_PointPillarsScatter(in_channels=feat_channels, output_shape=pseudo_image_dims)

    def forward(self, points: torch.Tensor) -> torch.Tensor:

        # List of points and coordinates for each batch
        # result_dict = {
        #         "points": valid_batch_non_nan_points,
        #         "voxel_coords": valid_batch_voxel_coords,
        #         "point_idxes": valid_point_idxes,
        #         "point_offsets": point_offsets
        #     }

        voxel_info_list = self.voxelizer(points)

        pseudoimage_lst = []
        pt_fea_lst = []
        out_bev_feats_lst = []
        out_bev_coors_lst = []

        for voxel_info_dict in voxel_info_list:
            points = voxel_info_dict['points']
            coordinates = voxel_info_dict['voxel_coords']
            # self.feature_net = DynamicPillarFeatureNet
           
            voxel_feats, voxel_coors, point_feats = self.feature_net(points, coordinates)
            # voxel_feats : torch.Size([NUM_non_empty_voxel, 32])
            # voxel_coors : torch.Size([NUM_non_empty_voxel, 3])   Example [ 0, 464,  99 ]
            # point_feats : torch.Size([ N_points , 32])
            pt_fea_lst.append(point_feats)

            if self.is_att:
                key_value_list = []
                cur_dev = voxel_coors.get_device()
                # [N, 2]
                bev_coors = voxel_coors[:,1:]
                # [N, 32]
                bev_feats = voxel_feats
                # [9, N, 2]
                shifted_index = shift_bev_grids(bev_coors, INDEX_SHIFT, self.pseudo_image_dims, cur_dev)

                for i, each_shift_index in enumerate(shifted_index):
                    select_ind = return_tensor_index(value = each_shift_index, t=bev_coors) # [N]
                    select_ind = torch.tensor(select_ind, device = cur_dev)
                    condition = (select_ind >= 0).unsqueeze(1).expand_as(bev_feats) # (N, CHANNEL=32)

                    tmp = bev_feats[select_ind] # (N, CHANNEL=32)
                    tmp= tmp.masked_fill(~condition, 0) # fill zero if select_ind  == -1
                    key_value_list.append(tmp)
                key = torch.stack(key_value_list) # [9, N , 32]
                key = key.permute(1, 0, 2) # (N, 9, CHANNEL=32)

                feat_query = bev_feats.unsqueeze(1) # (N, 1, CHANNEL=32)
                out, _ = self.atten_fusion(feat_query, key, key) # (N, 1, CHANNEL)
                out = out.squeeze(1)
                out_bev_feats = bev_feats + out

                pseudoimage = self.scatter(out_bev_feats, voxel_coors)
                pseudoimage_lst.append(pseudoimage)

                out_bev_feats_lst.append(out_bev_feats)
                out_bev_coors_lst.append(bev_coors)

                
            else: 
                # [N, 2]
                bev_coors = voxel_coors[:,1:]
                pseudoimage = self.scatter(voxel_feats, voxel_coors)
                pseudoimage_lst.append(pseudoimage)

                out_bev_feats_lst.append(voxel_feats)
                out_bev_coors_lst.append(bev_coors)
            
        # Concatenate the pseudoimages along the batch dimension
        return torch.cat(pseudoimage_lst, dim=0), voxel_info_list, pt_fea_lst, out_bev_feats_lst, out_bev_coors_lst


class lidar_DynamicEmbedder(nn.Module):
    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 feat_channels: int, mix = False, is_att = False) -> None:
        super().__init__()
        self.mix = mix
        self.is_att = is_att
        self.pseudo_image_dims = pseudo_image_dims
        
        self.voxelizer = DynamicVoxelizer(voxel_size=voxel_size,
                                          point_cloud_range=point_cloud_range)
        self.feature_net = Fusion_DynamicPillarFeatureNet(
            in_channels=3,
            feat_channels=(feat_channels,),  #  int(0.5*feat_channels), feat_channels,  # feat_channels,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            mode='avg',
            mix = self.mix) # 'avg'
        
        if self.is_att:
            # VERSION 1
            self.head_num = 2
            self.atten_fusion = torch.nn.MultiheadAttention(embed_dim=32, num_heads=self.head_num, dropout=0.1, batch_first=True)

        self.scatter = my_PointPillarsScatter(in_channels=feat_channels, output_shape=pseudo_image_dims)

    def forward(self, points: torch.Tensor) -> torch.Tensor:

        # List of points and coordinates for each batch
        # result_dict = {
        #         "points": valid_batch_non_nan_points,
        #         "voxel_coords": valid_batch_voxel_coords,
        #         "point_idxes": valid_point_idxes,
        #         "point_offsets": point_offsets
        #     }

        voxel_info_list = self.voxelizer(points)

        pseudoimage_lst = []
        pt_fea_lst = []
        out_bev_feats_lst = []
        out_bev_coors_lst = []

        for voxel_info_dict in voxel_info_list:
            points = voxel_info_dict['points']
            coordinates = voxel_info_dict['voxel_coords']
            # self.feature_net = DynamicPillarFeatureNet
           
            # [ 0, 113, 325],
            voxel_feats, voxel_coors, point_feats = self.feature_net(points, coordinates)
            # voxel_feats : torch.Size([NUM_non_empty_voxel, 32])
            # voxel_coors : torch.Size([NUM_non_empty_voxel, 3])   Example [ 0, 464,  99 ]
            # point_feats : torch.Size([ N_points , 32])
            pt_fea_lst.append(point_feats)

            if self.is_att:
                key_value_list = []
                cur_dev = voxel_coors.get_device()
                # [N, 2]
                bev_coors = voxel_coors[:,1:]
                # [N, 32]
                bev_feats = voxel_feats
                # [9, N, 2]
                shifted_index = shift_bev_grids(bev_coors, INDEX_SHIFT, self.pseudo_image_dims, cur_dev)
                # VERSION 1
                # ------------------------------------------------------------------------------------------------------
                for i, each_shift_index in enumerate(shifted_index):
                    select_ind = return_tensor_index(value = each_shift_index, t=bev_coors) # [N]
                    select_ind = torch.tensor(select_ind, device = cur_dev)
                    condition = (select_ind >= 0).unsqueeze(1).expand_as(bev_feats) # (N, CHANNEL=32)

                    tmp = bev_feats[select_ind] # (N, CHANNEL=32)
                    tmp= tmp.masked_fill(~condition, 0) # fill zero if select_ind  == -1
                    key_value_list.append(tmp)
                key = torch.stack(key_value_list) # [9, N , 32]
                key = key.permute(1, 0, 2) # (N, 9, CHANNEL=32)

                feat_query = bev_feats.unsqueeze(1) # (N, 1, CHANNEL=32)
                out, _ = self.atten_fusion(feat_query, key, key) # (N, 1, CHANNEL)
                out = out.squeeze(1)
                out_bev_feats = bev_feats + out
                pseudoimage = self.scatter(out_bev_feats, voxel_coors)
                pseudoimage_lst.append(pseudoimage)

                out_bev_feats_lst.append(out_bev_feats)
                out_bev_coors_lst.append(bev_coors)              
            else: 
                # [N, 2]
                bev_coors = voxel_coors[:,1:]
                pseudoimage = self.scatter(voxel_feats, voxel_coors)
                pseudoimage_lst.append(pseudoimage)

                out_bev_feats_lst.append(voxel_feats)
                out_bev_coors_lst.append(bev_coors)
            
        # Concatenate the pseudoimages along the batch dimension
        return torch.cat(pseudoimage_lst, dim=0), voxel_info_list, pt_fea_lst, out_bev_feats_lst, out_bev_coors_lst


class pillar_DynamicEmbedder(nn.Module):
    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 feat_channels: int, input_channels: int) -> None:
        super().__init__()

        self.pseudo_image_dims = pseudo_image_dims
        
        self.voxelizer = DynamicVoxelizer(voxel_size=voxel_size,
                                          point_cloud_range=point_cloud_range)
        self.feature_net = DynamicPillarFeatureNet(
            in_channels=input_channels,
            feat_channels=(feat_channels,),  #  int(0.5*feat_channels), feat_channels,  # feat_channels,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            mode='avg') # 'avg'
        
        self.scatter = my_PointPillarsScatter(in_channels=feat_channels, output_shape=pseudo_image_dims)

    def forward(self, points: torch.Tensor) -> torch.Tensor:

        # List of points and coordinates for each batch
        # result_dict = {
        #         "points": valid_batch_non_nan_points,
        #         "voxel_coords": valid_batch_voxel_coords,
        #         "point_idxes": valid_point_idxes,
        #         "point_offsets": point_offsets
        #     }

        voxel_info_list = self.voxelizer(points)

        pseudoimage_lst = []
        pt_fea_lst = []
        out_bev_feats_lst = []
        out_bev_coors_lst = []

        for voxel_info_dict in voxel_info_list:
            points = voxel_info_dict['points']
            coordinates = voxel_info_dict['voxel_coords']
            # self.feature_net = DynamicPillarFeatureNet
           
            # [ 0, 113, 325],
            voxel_feats, voxel_coors, point_feats = self.feature_net(points, coordinates)
            # voxel_feats : torch.Size([NUM_non_empty_voxel, 32])
            # voxel_coors : torch.Size([NUM_non_empty_voxel, 3])   Example [ 0, 464,  99 ]
            # point_feats : torch.Size([ N_points , 32])
            pt_fea_lst.append(point_feats)

                   
            # [N, 2]
            bev_coors = voxel_coors[:,1:]
            pseudoimage = self.scatter(voxel_feats, voxel_coors)
            pseudoimage_lst.append(pseudoimage)

            out_bev_feats_lst.append(voxel_feats)
            out_bev_coors_lst.append(bev_coors)
            
        # Concatenate the pseudoimages along the batch dimension
        return torch.cat(pseudoimage_lst, dim=0), voxel_info_list, pt_fea_lst, out_bev_feats_lst, out_bev_coors_lst


class old_lidar_DynamicEmbedder(nn.Module):
    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 feat_channels: int) -> None:
        super().__init__()
        self.voxelizer = DynamicVoxelizer(voxel_size=voxel_size,
                                          point_cloud_range=point_cloud_range)
        self.feature_net = Fusion_DynamicPillarFeatureNet(
            in_channels=3,
            feat_channels=( int(0.5*feat_channels), feat_channels,),
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            mode='max',
            mix = True) # 'avg'
        self.scatter = PointPillarsScatter(in_channels=feat_channels,
                                           output_shape=pseudo_image_dims)

    def forward(self, points: torch.Tensor) -> torch.Tensor:

        # List of points and coordinates for each batch
        # result_dict = {
        #         "points": valid_batch_non_nan_points,
        #         "voxel_coords": valid_batch_voxel_coords,
        #         "point_idxes": valid_point_idxes,
        #         "point_offsets": point_offsets
        #     }

        voxel_info_list = self.voxelizer(points)

        pseudoimage_lst = []
        pt_fea_lst = []

        for voxel_info_dict in voxel_info_list:
            points = voxel_info_dict['points']
            coordinates = voxel_info_dict['voxel_coords']
            # self.feature_net = DynamicPillarFeatureNet
           
            voxel_feats, voxel_coors, point_feats = self.feature_net(points, coordinates)
            # voxel_feats : torch.Size([NUM_non_empty_voxel, 32])
            # voxel_coors : torch.Size([NUM_non_empty_voxel, 3])   Example [ 0, 464,  99 ]
            # point_feats : torch.Size([ N_points , 32])
         
            pseudoimage = self.scatter(voxel_feats, voxel_coors)
            pseudoimage_lst.append(pseudoimage)
            pt_fea_lst.append(point_feats)
        # Concatenate the pseudoimages along the batch dimension
        return torch.cat(pseudoimage_lst, dim=0), voxel_info_list, pt_fea_lst


class old_radar_DynamicEmbedder(nn.Module):
    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 feat_channels: int) -> None:
        super().__init__()
        self.voxelizer = DynamicVoxelizer(voxel_size=voxel_size,
                                          point_cloud_range=point_cloud_range)
        self.feature_net = Fusion_DynamicPillarFeatureNet(
            in_channels=5,
            feat_channels=( int(0.5*feat_channels), feat_channels,),
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            mode='max',
            mix = True) # 'avg'
        self.scatter = PointPillarsScatter(in_channels=feat_channels,
                                           output_shape=pseudo_image_dims)

    def forward(self, points: torch.Tensor) -> torch.Tensor:

        # List of points and coordinates for each batch
        # result_dict = {
        #         "points": valid_batch_non_nan_points,
        #         "voxel_coords": valid_batch_voxel_coords,
        #         "point_idxes": valid_point_idxes,
        #         "point_offsets": point_offsets
        #     }

        voxel_info_list = self.voxelizer(points)

        pseudoimage_lst = []
        pt_fea_lst = []

        for voxel_info_dict in voxel_info_list:
            points = voxel_info_dict['points']
            coordinates = voxel_info_dict['voxel_coords']
            # self.feature_net = DynamicPillarFeatureNet
            # point_feats : torch.Size([ N , 32])
            voxel_feats, voxel_coors, point_feats = self.feature_net(points, coordinates)
            pseudoimage = self.scatter(voxel_feats, voxel_coors)
            pseudoimage_lst.append(pseudoimage)
            pt_fea_lst.append(point_feats)

        # Concatenate the pseudoimages along the batch dimension
        return torch.cat(pseudoimage_lst, dim=0), voxel_info_list, pt_fea_lst

