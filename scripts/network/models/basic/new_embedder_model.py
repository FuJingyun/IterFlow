import torch
import torch.nn as nn

from .make_voxels import  DynamicVoxelizer  # HardVoxelizer,
from .process_voxels import Fusion_DynamicPillarFeatureNet # DynamicPillarFeatureNet # PillarFeatureNet,  
from .scatter import PointPillarsScatter
from .bev_shift import shift_bev_grids, return_tensor_index
# import torch_scatter

import spconv as spconv_core
spconv_core.constants.SPCONV_ALLOW_TF32 = True
import spconv.pytorch as spconv


class new_radar_DynamicEmbedder(nn.Module):
    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 feat_channels: int, mix = False) -> None:
        super().__init__()
        self.mix = mix
        self.pseudo_image_dims = pseudo_image_dims
        self.sparse_3d_dims=[ pseudo_image_dims[0],  pseudo_image_dims[1] , 1]
        
        self.voxelizer = DynamicVoxelizer(voxel_size=voxel_size,
                                          point_cloud_range=point_cloud_range)
        self.feature_net = Fusion_DynamicPillarFeatureNet(
            in_channels=6,
            feat_channels=(feat_channels,),  #  int(0.5*feat_channels), feat_channels,  # feat_channels,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            mode='avg',
            mix = self.mix) # 'avg'
        

        self.scatter = PointPillarsScatter(in_channels=feat_channels, output_shape=pseudo_image_dims)

    def forward(self, points: torch.Tensor) -> torch.Tensor:

        voxel_info_list = self.voxelizer(points)

        # pseudoimage_lst = []
        pt_fea_lst = []
        out_feats_lst = []
        out_coors_lst = []
        voxel_feats_list_batch = []
        voxel_coors_list_batch = []

        for batch_index, voxel_info_dict in enumerate(voxel_info_list):
            points = voxel_info_dict['points']
            coordinates = voxel_info_dict['voxel_coords']
            # self.feature_net = DynamicPillarFeatureNet
           
            voxel_feats, voxel_coors, point_feats = self.feature_net(points, coordinates)
            # voxel_feats : torch.Size([NUM_non_empty_voxel, 32])
            # voxel_coors : torch.Size([NUM_non_empty_voxel, 3])   Example [ 0, 464,  99 ]
            # point_feats : torch.Size([ N_points , 32])
            pt_fea_lst.append(point_feats)
            out_feats_lst.append(voxel_feats)
            out_coors_lst.append(voxel_coors)

            batch_indices = torch.full((voxel_coors.size(0), 1), batch_index, dtype=torch.long, device=voxel_coors.device)
            # for 3d
            # voxel_coors_batch = torch.cat([batch_indices, voxel_coors[:, [2, 1, 0]]], dim=1)
            # for 2d
            voxel_coors_batch = torch.cat([batch_indices, voxel_coors[:, [2, 1]]], dim=1)
    
            voxel_feats_list_batch.append(voxel_feats)
            voxel_coors_list_batch.append(voxel_coors_batch)

        voxel_feats_sp = torch.cat(voxel_feats_list_batch, dim=0)
        coors_batch_sp = torch.cat(voxel_coors_list_batch, dim=0).to(dtype=torch.int32)
        # sparse_tensor_3d = spconv.SparseConvTensor(voxel_feats_sp.contiguous(), coors_batch_sp.contiguous(), self.sparse_3d_dims, int(batch_index + 1))

        sparse_tensor_2d = spconv.SparseConvTensor(voxel_feats_sp.contiguous(), coors_batch_sp.contiguous(), self.pseudo_image_dims, int(batch_index + 1))
         
        # Concatenate the pseudoimages along the batch dimension
        # return sparse_tensor_3d,voxel_info_list, pt_fea_lst, out_feats_lst, out_coors_lst
        return sparse_tensor_2d, voxel_info_list, pt_fea_lst, out_feats_lst, out_coors_lst





class new_lidar_DynamicEmbedder(nn.Module):
    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 feat_channels: int, mix = False) -> None:
        super().__init__()
        self.mix = mix
        self.pseudo_image_dims = pseudo_image_dims
        self.sparse_3d_dims=[ pseudo_image_dims[0], pseudo_image_dims[1], 1]
        
        self.voxelizer = DynamicVoxelizer(voxel_size=voxel_size,
                                          point_cloud_range=point_cloud_range)
        self.feature_net = Fusion_DynamicPillarFeatureNet(
            in_channels=3,
            feat_channels=(feat_channels,),  #  int(0.5*feat_channels), feat_channels,  # feat_channels,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            mode='avg',
            mix = self.mix) # 'avg'
        

        self.scatter = PointPillarsScatter(in_channels=feat_channels, output_shape=pseudo_image_dims)

    def forward(self, points: torch.Tensor) -> torch.Tensor:

        voxel_info_list = self.voxelizer(points)

        # pseudoimage_lst = []
        pt_fea_lst = []
        out_feats_lst = []
        out_coors_lst = []
        voxel_feats_list_batch = []
        voxel_coors_list_batch = []

        for batch_index, voxel_info_dict in enumerate(voxel_info_list):
            points = voxel_info_dict['points']
            coordinates = voxel_info_dict['voxel_coords']
            # self.feature_net = DynamicPillarFeatureNet
           
            voxel_feats, voxel_coors, point_feats = self.feature_net(points, coordinates)
            # voxel_feats : torch.Size([NUM_non_empty_voxel, 32])
            # voxel_coors : torch.Size([NUM_non_empty_voxel, 3])   Example [ 0, 464,  99 ]
            # point_feats : torch.Size([ N_points , 32])
            pt_fea_lst.append(point_feats)
            out_feats_lst.append(voxel_feats)
            out_coors_lst.append(voxel_coors)

            batch_indices = torch.full((voxel_coors.size(0), 1), batch_index, dtype=torch.long, device=voxel_coors.device)
            # for 3d
            # voxel_coors_batch = torch.cat([batch_indices, voxel_coors[:, [2, 1, 0]]], dim=1)
            # for 2d
            voxel_coors_batch = torch.cat([batch_indices, voxel_coors[:, [2, 1]]], dim=1)
        
            voxel_feats_list_batch.append(voxel_feats)
            voxel_coors_list_batch.append(voxel_coors_batch)

        voxel_feats_sp = torch.cat(voxel_feats_list_batch, dim=0)
        coors_batch_sp = torch.cat(voxel_coors_list_batch, dim=0).to(dtype=torch.int32)
        #  sparse_tensor_3d = spconv.SparseConvTensor(voxel_feats_sp.contiguous(), coors_batch_sp.contiguous(), self.sparse_3d_dims, int(batch_index + 1))

        sparse_tensor_2d = spconv.SparseConvTensor(voxel_feats_sp.contiguous(), coors_batch_sp.contiguous(), self.pseudo_image_dims, int(batch_index + 1))

        # return sparse_tensor_3d, voxel_info_list, pt_fea_lst, out_feats_lst, out_coors_lst

        return sparse_tensor_2d, voxel_info_list, pt_fea_lst, out_feats_lst, out_coors_lst

