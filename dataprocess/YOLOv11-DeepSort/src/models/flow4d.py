"""
This file is copied from:
https://github.com/dgist-cvlab/Flow4D

Modified by Siyi Li (UniFlow) – 2025-12-28
Changes:
- Added support for larger backbone variants.
"""

import torch.nn as nn
import dztimer, torch

from .basic import wrap_batch_pcs
from .basic.flow4d_module import DynamicEmbedder_4D
from .basic.flow4d_module import Network_4D, Seperate_to_3D, Point_head

from dataclasses import dataclass
from typing import Optional, Tuple

def _compute_grid_feature_size(point_cloud_range, voxel_size):
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    vx, vy, vz = voxel_size

    nx = int(round((x_max - x_min) / vx))
    ny = int(round((y_max - y_min) / vy))
    nz = int(round((z_max - z_min) / vz))
    return (ny, nx, nz)

@dataclass
class BackboneConfig:
    model_size: Optional[int] = None
    point_out: Optional[int] = None
    voxel_out: Optional[int] = None
    head_hidden: Optional[int] = None

    def resolve(self):
        default_point = self.model_size if self.model_size is not None else 16
        default_voxel = (2 * self.model_size) if self.model_size is not None else 16

        point_out = self.point_out if self.point_out is not None else default_point
        voxel_out = self.voxel_out if self.voxel_out is not None else default_voxel

        inferred_hidden = max(32, (point_out + voxel_out) // 2)
        head_hidden = self.head_hidden if self.head_hidden is not None else inferred_hidden
        return point_out, voxel_out, head_hidden

BACKBONES = {
    "base":  BackboneConfig(model_size=16, point_out=16, voxel_out=16),
    "large": BackboneConfig(model_size=32, point_out=32, voxel_out=32),
    "xl":    BackboneConfig(model_size=48, point_out=48, voxel_out=48),
}

class Flow4D(nn.Module):
    def __init__(self, voxel_size = [0.2, 0.2, 0.2],
                 point_cloud_range = [-51.2, -51.2, -2.2, 51.2, 51.2, 4.2],
                 grid_feature_size = [512, 512, 32], # We have to keep this to make the old checkpoints valid
                 num_frames = 5,
                 backbone = 'base'):
        super().__init__()

        cfg = BACKBONES[backbone]  
        point_out, voxel_out, head_hidden = cfg.resolve()
        self.num_frames = num_frames
        print("Using Backbone: ", backbone)
        grid_feature_size = _compute_grid_feature_size(point_cloud_range, voxel_size)
        print('voxel_size = {}, pseudo_dims = {}, input_num_frames = {}'.format(
            voxel_size, grid_feature_size, self.num_frames))

        self.embedder_4D = DynamicEmbedder_4D(
            voxel_size=voxel_size,
            pseudo_image_dims=[*grid_feature_size, num_frames],
            point_cloud_range=point_cloud_range,
            feat_channels=point_out,
        )

        self.network_4D = Network_4D(
            in_channel=point_out,
            out_channel=voxel_out,
            model_size=cfg.model_size if cfg.model_size is not None else 16,  
        )

        self.seperate_feat = Seperate_to_3D(num_frames)
        self.pointhead_3D = Point_head(
            voxel_feat_dim=voxel_out,
            point_feat_dim=point_out,
            hidden=head_hidden,
        )
        
        self.timer = dztimer.Timing()
        self.timer.start("Total")

    def forward(self, batch):
        """
        input: using the batch from dataloader, which is a dict
               Detail: [pc0, pc1, pose0, pose1]
        output: the predicted flow, pose_flow, and the valid point index of pc0
        """

        self.timer[0].start("Data Preprocess")
        pcs_dict = wrap_batch_pcs(batch, self.num_frames)
        self.timer[0].stop()

        self.timer[1].start("4D_voxelization")
        dict_4d = self.embedder_4D(pcs_dict)
        pc01_tesnor_4d = dict_4d['4d_tensor']
        pch1_3dvoxel_infos_lst = dict_4d['pch1_3dvoxel_infos_lst']
        pc0_3dvoxel_infos_lst =dict_4d['pc0_3dvoxel_infos_lst']

        pc0_point_feats_lst =dict_4d['pc0_point_feats_lst']
        pc0_num_voxels = dict_4d['pc0_num_voxels']

        pc1_3dvoxel_infos_lst =dict_4d['pc1_3dvoxel_infos_lst']
        self.timer[1].stop()

        self.timer[2].start("4D_backbone")
        pc_all_output_4d = self.network_4D(pc01_tesnor_4d) #all = past, current, next 다 합친것
        self.timer[2].stop()

        self.timer[3].start("4D pc01 to 3D pc0")
        pc0_last = self.seperate_feat(pc_all_output_4d)
        assert pc0_last.features.shape[0] == pc0_num_voxels, 'voxel number mismatch'
        self.timer[3].stop()

        self.timer[4].start("3D_sparsetensor_to_point and head")
        flows = self.pointhead_3D(pc0_last, pc0_3dvoxel_infos_lst, pc0_point_feats_lst)
        self.timer[4].stop()

        model_res = {
            "flow": flows, 
            'pose_flow': pcs_dict['pose_flows'], 

            "pc0_valid_point_idxes": [e["point_idxes"] for e in pc0_3dvoxel_infos_lst], 
            "pc0_points_lst": [e["points"] for e in pc0_3dvoxel_infos_lst] , 
            
            "pc1_valid_point_idxes": [e["point_idxes"] for e in pc1_3dvoxel_infos_lst],
            "pc1_points_lst": [e["points"] for e in pc1_3dvoxel_infos_lst],

            'pch1_valid_point_idxes': [e["point_idxes"] for e in pch1_3dvoxel_infos_lst] if pch1_3dvoxel_infos_lst != None else None,
        }
        return model_res