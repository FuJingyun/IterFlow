import torch
import torch.nn as nn

from .make_voxels import HardVoxelizer, DynamicVoxelizer
from .process_voxels import PillarFeatureNet, DynamicPillarFeatureNet
from .scatter import PointPillarsScatter, my_PointPillarsScatter


import torch.nn.functional as F


class HardEmbedder(nn.Module):

    def __init__(self,
                 voxel_size=(0.2, 0.2, 4),
                 pseudo_image_dims=(350, 350),
                 point_cloud_range=(-35, -35, -3, 35, 35, 1),
                 max_points_per_voxel=128,
                 feat_channels=64) -> None:
        super().__init__()
        self.voxelizer = HardVoxelizer(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_points_per_voxel=max_points_per_voxel)
        self.feature_net = PillarFeatureNet(
            in_channels=3,
            feat_channels=(feat_channels, ),
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size)
        self.scatter = PointPillarsScatter(in_channels=feat_channels,
                                           output_shape=pseudo_image_dims)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            points,
            torch.Tensor), f"points must be a torch.Tensor, got {type(points)}"

        output_voxels, output_voxel_coords, points_per_voxel = self.voxelizer(
            points)
        output_features = self.feature_net(output_voxels, points_per_voxel,
                                           output_voxel_coords)
        pseudoimage = self.scatter(output_features, output_voxel_coords)

        return pseudoimage



class lidar_DynamicEmbedder(nn.Module):
    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 feat_channels: int) -> None:
        super().__init__()
        self.voxelizer = DynamicVoxelizer(voxel_size=voxel_size,
                                          point_cloud_range=point_cloud_range)
        self.feature_net = DynamicPillarFeatureNet(
            in_channels=3,
            feat_channels=( 0.5*feat_channels, feat_channels,),
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            mode='mix') # 'avg'
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
        for voxel_info_dict in voxel_info_list:
            points = voxel_info_dict['points']
            coordinates = voxel_info_dict['voxel_coords']
            # self.feature_net = DynamicPillarFeatureNet
            # point_feats : torch.Size([ N , 32])
            voxel_feats, voxel_coors, point_feats = self.feature_net(points, coordinates)
            pseudoimage = self.scatter(voxel_feats, voxel_coors)
            pseudoimage_lst.append(pseudoimage)
        # Concatenate the pseudoimages along the batch dimension
        return torch.cat(pseudoimage_lst, dim=0), voxel_info_list


class radar_DynamicEmbedder(nn.Module):
    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 feat_channels: int) -> None:
        super().__init__()
        self.voxelizer = DynamicVoxelizer(voxel_size=voxel_size,
                                          point_cloud_range=point_cloud_range)
        self.feature_net = DynamicPillarFeatureNet(
            in_channels=5,
            feat_channels=( 0.5*feat_channels, feat_channels,),
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            mode='mix') # 'avg'
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
        for voxel_info_dict in voxel_info_list:
            points = voxel_info_dict['points']
            coordinates = voxel_info_dict['voxel_coords']
            # self.feature_net = DynamicPillarFeatureNet
            # point_feats : torch.Size([ N , 32])
            voxel_feats, voxel_coors, point_feats = self.feature_net(points, coordinates)
            pseudoimage = self.scatter(voxel_feats, voxel_coors)
            pseudoimage_lst.append(pseudoimage)
        # Concatenate the pseudoimages along the batch dimension
        return torch.cat(pseudoimage_lst, dim=0), voxel_info_list


class old_DynamicEmbedder(nn.Module):

    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 feat_channels: int) -> None:
        super().__init__()
        self.voxelizer = DynamicVoxelizer(voxel_size=voxel_size,
                                          point_cloud_range=point_cloud_range)
        self.feature_net = DynamicPillarFeatureNet(
            in_channels=3,  # 3
            feat_channels=(feat_channels, ),
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            mode='avg')
        self.scatter = my_PointPillarsScatter(in_channels=feat_channels,
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
        for voxel_info_dict in voxel_info_list:
            # points = [x,y,z]
            points = voxel_info_dict['points']
            # coordinates [0, 452, 80]
            # Z Y X
            coordinates = voxel_info_dict['voxel_coords']

            # point_feats : torch.Size([ N , 32])
            # voxel_coors : [  0, 460,  84] Z Y X
            voxel_feats, voxel_coors, point_feats = self.feature_net(points, coordinates)

            pseudoimage = self.scatter(voxel_feats, voxel_coors)
            pseudoimage_lst.append(pseudoimage)
        
        # Concatenate the pseudoimages along the batch dimension
        return torch.cat(pseudoimage_lst, dim=0), voxel_info_list



class DynamicEmbedder(nn.Module):

    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 feat_channels: int) -> None:
        super().__init__()
        self.voxelizer = DynamicVoxelizer(voxel_size=voxel_size,
                                          point_cloud_range=point_cloud_range)
        self.feature_net = DynamicPillarFeatureNet(
            in_channels=3,
            feat_channels=(feat_channels, ),
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            mode='avg')
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
            # # point_feats : torch.Size([ N , 32])
            voxel_feats, voxel_coors, point_feats = self.feature_net(points, coordinates)
            pseudoimage = self.scatter(voxel_feats, voxel_coors)
            pseudoimage_lst.append(pseudoimage)
            pt_fea_lst.append(point_feats)
        # Concatenate the pseudoimages along the batch dimension
        return torch.cat(pseudoimage_lst, dim=0), voxel_info_list, pt_fea_lst



class old_vod_DynamicEmbedder(nn.Module):

    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 feat_channels: int) -> None:
        super().__init__()
        self.voxelizer = DynamicVoxelizer(voxel_size=voxel_size,
                                          point_cloud_range=point_cloud_range)
        self.feature_net = DynamicPillarFeatureNet(
            # 
            in_channels=6, # 5 for first version of L+R input, L output
            feat_channels=(feat_channels, ),
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            mode='avg')
        self.scatter = my_PointPillarsScatter(in_channels=feat_channels,
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

        for voxel_info_dict in voxel_info_list:
            points = voxel_info_dict['points']
            coordinates = voxel_info_dict['voxel_coords']
            # self.feature_net = DynamicPillarFeatureNet
            voxel_feats, voxel_coors, point_feats = self.feature_net(points, coordinates)
            pseudoimage = self.scatter(voxel_feats, voxel_coors)
            pseudoimage_lst.append(pseudoimage)
        # Concatenate the pseudoimages along the batch dimension
        return torch.cat(pseudoimage_lst, dim=0), voxel_info_list

class vod_DynamicEmbedder(nn.Module):

    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 feat_channels: int) -> None:
        super().__init__()
        self.voxelizer = DynamicVoxelizer(voxel_size=voxel_size,
                                          point_cloud_range=point_cloud_range)
        self.feature_net = DynamicPillarFeatureNet(
            # 
            in_channels=6, # 5 for first version of L+R input, L output
            feat_channels=(feat_channels, ),
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            mode='avg')
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
            pseudoimage = self.scatter(voxel_feats, voxel_coors)
            pseudoimage_lst.append(pseudoimage)
            pt_fea_lst.append(point_feats)
        # Concatenate the pseudoimages along the batch dimension
        return torch.cat(pseudoimage_lst, dim=0), voxel_info_list, pt_fea_lst