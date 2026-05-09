
"""
# Created: 2023-07-18 15:08
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""

import torch.nn as nn
import dztimer, torch
import sys

from .basic import cal_pose0to1
from .basic.PT_util.corr import BQ_CorrBlock
from .basic.PT_util.update import UpdateBlock
from .basic.PT_util.flot.graph import Graph
# from .basic.PT_util.extractor import FlotEncoder, sinkhorn
# from scripts.network.models.basic.PT_util.flot.gconv import SetConv
from scripts.network.models.basic.PT_util.flot.graph import Graph

import copy


# import torch.nn as nn
import torch
import numpy as np
import os
import torch.nn.functional as F
from cmflow_utils.model_utils import *
from cmflow_utils import *
# from .basic.radarflow_util import Conv1d, EncoderBlock, simple_EncoderBlock


class IterFlow(nn.Module):
    def __init__(self, 
                num_points = 256,
                eval_only = False
                ):
        super().__init__()
        self.npoints = num_points
        self.eval_only = eval_only

        self.num_neighbors = 8
        self.hidden_dim = 64 # 64
        self.context_dim = 64 # 64
        self.num_iters = 12 # 12


        ## multi-scale set feature abstraction 
        sa_radius = [2.0, 4.0, 8.0, 16.0]
        sa_nsamples = [4, 8, 16, 32]
        sa_mlps = [16, 16, 32]
        sa_mlp2s = [32, 32, 32]


        self.mse_layer = MultiScaleEncoder(sa_radius, sa_nsamples, in_channel=2, \
                                         mlp = sa_mlps, mlp2 = sa_mlp2s)
    
        self.mse_layer2 = MultiScaleEncoder(sa_radius, sa_nsamples, in_channel=2, \
                                            mlp = sa_mlps, mlp2 = sa_mlp2s)

        self.corr_block = BQ_CorrBlock()                    
        self.update_block = UpdateBlock(input_dim=self.hidden_dim+64, hidden_dim = self.hidden_dim)

    
        self.timer = dztimer.Timing()
        self.timer.start("Total")
         
    # My edit: Only for evaluation
    def forward(self,batch):             
    # def forward(self, pc1, pc2, feature1, feature2, label_m, mode):
        '''
        pc1: B 3 N
        pc2: B 3 N
        feature1: B 3 N
        feature2: B 3 N
        # pc1  N×5  Source radar point clouds (x, y, z, RCS, doppler velocity).
        # pc2  M×5  Target radar point clouds (x, y, z, RCS, doppler velocity).
        pos_1 = data_1[:,0:3]
        pos_2 = data_2[:,0:3]
        feature_1 = data_1[:,[4,3,3]]
        feature_2 = data_2[:,[4,3,3]] 
        '''
        self.timer[0].start("Data Preprocess")
        batch_sizes = len(batch["pose0"])

        radar_pose_flows = []
        radar_pc0_points_lst = []
        radar_pc1_points_lst = []
        radar_pc0_valid_point_idxes = []
        radar_pc1_valid_point_idxes = []

        for batch_id in range(batch_sizes):
            radar_pc0 = batch["radar_pc0"][batch_id] # N 3
            radar_pc1 = batch["radar_pc1"][batch_id] # N 3

            self.timer[0][0].start("pose")
            with torch.no_grad():
                if 'ego_motion' in batch:
                    pose_0to1 = batch['ego_motion'][batch_id]
                else:
                    pose_0to1 = cal_pose0to1(batch["pose0"][batch_id], batch["pose1"][batch_id])
            self.timer[0][0].stop()

            self.timer[0][1].start("transform")
            transform_radar_pc0_xyz = radar_pc0[:,:3]  @ pose_0to1[:3, :3].T + pose_0to1[:3, 3]
            transform_radar_pc0 = copy.deepcopy(radar_pc0)
            transform_radar_pc0[:,:3] = transform_radar_pc0_xyz
            self.timer[0][1].stop()

            radar_pose_flows.append(transform_radar_pc0_xyz - radar_pc0[:,:3])

            valid_point_idxes_pc0 = torch.arange(radar_pc0.shape[0],
                                             device=radar_pc0.device)
            valid_point_idxes_pc1 = torch.arange(radar_pc1.shape[0],
                                             device=radar_pc1.device)
            
            radar_pc0_points_lst.append(radar_pc0)
            radar_pc1_points_lst.append(radar_pc1)
            radar_pc0_valid_point_idxes.append(valid_point_idxes_pc0)
            radar_pc1_valid_point_idxes.append(valid_point_idxes_pc1)

        pc1 = batch['radar_pc0'].transpose(1, 2).contiguous() # B 3 N
        pc2 = batch['radar_pc1'].transpose(1, 2).contiguous() # B 3 N
        feature1 = batch['radar_ft0'].transpose(1, 2).contiguous() # B 3 N
        feature2 = batch['radar_ft1'].transpose(1, 2).contiguous() # B 3 N

        fmap1 = self.mse_layer(pc1,feature1) # B 128 N
        fmap2 = self.mse_layer(pc2,feature2) # B 128 N
        fct1 = self.mse_layer2(pc1,feature1) # B 128 N

        pc1 = pc1.transpose(1, 2).contiguous()
        pc2 = pc2.transpose(1, 2).contiguous()

        # correlation matrix
        self.corr_block.init_module(fmap1, fmap2)
        graph_context = Graph.construct_graph(pc1, self.num_neighbors)

        net, inp = torch.split(fct1, [self.hidden_dim, self.context_dim], dim=1)
        # net ：hidden state ht-1 from context feature
        net = torch.tanh(net)
        inp = torch.relu(inp)

        flow_predictions = []
        coords1, coords2 = pc1, pc2

        for itr in range(self.num_iters):
            coords2 = coords2.detach()
            corr = self.corr_block(coords=coords2, xyz2 = pc2)
            flow = coords2 - coords1
            net, delta_flow = self.update_block(net, inp, corr, flow, graph_context)
            coords2 = coords2 + delta_flow
            flow_predictions.append(coords2 - coords1)
        

        radar_flow = flow_predictions[-1]
        model_res = {

            "pt_flow" : flow_predictions,

            "radar_flow": radar_flow,
            'radar_pose_flow': radar_pose_flows,

            "radar_pc0_valid_point_idxes": radar_pc0_valid_point_idxes,
            "radar_pc0_points_lst": radar_pc0_points_lst,
            
            "radar_pc1_valid_point_idxes": radar_pc1_valid_point_idxes,
            "radar_pc1_points_lst": radar_pc1_points_lst,
        }
        return model_res

    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("model.") :]: v for k, v in ckpt.items() if k.startswith("model.")
        }
        print("\nLoading... model weight from: ", ckpt_path, "\n")
        return self.load_state_dict(state_dict=state_dict, strict=False)