import torch
import torch.nn as nn
from typing import List, Tuple, Dict
import copy

SPLIT_BATCH_SIZE = 512


class Cascade_fb_Seg(nn.Module):
    def __init__(self, stat_thres: float = 0.5):
        super().__init__()

        self.flow_encoder = nn.Linear(3, 32)
        self.flow_map_encoder = nn.Linear(64, 32)
        self.pc_encoder = nn.Linear(64, 32)

        self.dy_layer1 =nn.Sequential(
                    nn.Linear(64, 32),
                    nn.BatchNorm1d(32, eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True))
        
        self.dy_layer2 =nn.Sequential(
                    nn.Linear(96, 32), nn.GELU(),
                    nn.Linear(32, 16), nn.GELU(),
                    nn.Linear(16, 1), nn.Sigmoid()
                    )
        
        self.stat_thres = stat_thres


    def seg_single(self, pc0_map: torch.Tensor, # torch.Size([64, 512, 512])
                       flow_map: torch.Tensor,  # torch.Size([64, 512, 512])
                       lidar_voxel_coords: torch.Tensor, # torch.Size([13216, 3])
                       lidar_flow: torch.Tensor, # torch.Size([13216, 3])
                       lidar_pc0_fea: torch.Tensor,
                       radar_voxel_coords: torch.Tensor, # torch.Size([13216, 3])
                       radar_flow: torch.Tensor, # torch.Size([13216, 3])
                       radar_pc0_fea: torch.Tensor
                       ) -> torch.Tensor:
        lidar_voxel_coords = lidar_voxel_coords.long()
        radar_voxel_coords = radar_voxel_coords.long()

        # [N, 64]
        lidar_pc0_map_vectors = pc0_map[:, lidar_voxel_coords[:, 1], lidar_voxel_coords[:, 2]].T
        radar_pc0_map_vectors = pc0_map[:, radar_voxel_coords[:, 1], radar_voxel_coords[:, 2]].T
        pc0_map_vectors = torch.cat((lidar_pc0_map_vectors, radar_pc0_map_vectors), 0)
        # [N, 64] -> [N, 32]
        pc_map_fea = self.pc_encoder(pc0_map_vectors)

        # [N, 64]
        lidar_flow_map_vectors = flow_map[:, lidar_voxel_coords[:, 1], lidar_voxel_coords[:, 2]].T
        radar_flow_map_vectors = flow_map[:, radar_voxel_coords[:, 1], radar_voxel_coords[:, 2]].T
        flow_map_vectors = torch.cat((lidar_flow_map_vectors, radar_flow_map_vectors), 0)
        # [N, 64] -> [N, 32]
        flow_map_fea = self.flow_map_encoder(flow_map_vectors)


        # pc0_fea.requires_grad = True
        # flow.requires_grad = True
        # flow_map_vectors.requires_grad = True
        # pc0_map_vectors.requires_grad = True

        # [N, 3] -> [N, 32]
        flow = torch.cat((lidar_flow, radar_flow), 0)
        flow_fea = self.flow_encoder(flow)

        pc0_fea = torch.cat((lidar_pc0_fea, radar_pc0_fea), 0)

        # Cascade
        fea_layer1 = self.dy_layer1(torch.cat([flow_fea,  pc_map_fea], dim=1))
        dynamic_score = self.dy_layer2(torch.cat([fea_layer1,  flow_map_fea, pc0_fea], dim=1))

        # return dynamic_score.squeeze(1),masked_flow
        return dynamic_score.squeeze(1)
    

    def forward(
            self, pc0_map: torch.Tensor,   # [B 64 512 512]
            flow_map: torch.Tensor,  # [B 64 512 512]
            voxelizer_infos: List[Dict[str, torch.Tensor]],
            pc0_point_fea: List[torch.Tensor],                        
            flows: List[torch.Tensor],
            radar_voxelizer_infos: List[Dict[str, torch.Tensor]],
            radar_pc0_point_fea: List[torch.Tensor],                        
            radar_flows: List[torch.Tensor]) -> List[torch.Tensor]:

        score_results = []
        batch_size = len(flows)
        # flow = []

        for batch_id in range(batch_size):
            cur_pc0_map = pc0_map[batch_id]  
            cur_flow_map = flow_map[batch_id]  
            # LiDAR
            voxel_coords = voxelizer_infos[batch_id]["voxel_coords"] 
            cur_flow = flows[batch_id] 
            cur_pc0_fea = pc0_point_fea[batch_id]
            # Radar
            radar_voxel_coords = radar_voxelizer_infos[batch_id]["voxel_coords"] 
            radar_cur_flow = radar_flows[batch_id] 
            radar_cur_pc0_fea = radar_pc0_point_fea[batch_id]
            # temp_score,t = self.seg_single(cur_pc0_map, cur_flow_map, voxel_coords, cur_flow, cur_pc0_fea)
            temp_score = self.seg_single(cur_pc0_map, cur_flow_map, \
                                         voxel_coords, cur_flow, cur_pc0_fea, \
                                         radar_voxel_coords, radar_cur_flow, radar_cur_pc0_fea  )
            score_results.append(temp_score)
            # flow.append(t)

        # return score_results,flow
        return score_results
    



class FB_Seg(nn.Module):
    def __init__(self, stat_thres: float = 0.5):
        super().__init__()

        # self.dy_layer1 =nn.Sequential(
        #             nn.Linear(128, 64),
        #             nn.BatchNorm1d(32, eps=1e-3, momentum=0.01),
        #             nn.ReLU(inplace=True))

        self.linear_layer = nn.Linear(128, 64)


        self.dy_layer =nn.Sequential(
                    nn.Linear(128, 64), nn.GELU(),
                    nn.Linear(64, 32), nn.GELU(),
                    nn.Linear(32, 16), nn.GELU(), 
                    nn.Linear(16, 1), nn.Sigmoid()
                    )
        
    def seg_single(self, pc0_map: torch.Tensor, # torch.Size([64, 512, 512])
                        pc1_map: torch.Tensor, # torch.Size([64, 512, 512])
                       flow_map: torch.Tensor,  # torch.Size([64, 512, 512])
                       lidar_voxel_coords: torch.Tensor, # torch.Size([13216, 3])
                       radar_voxel_coords: torch.Tensor, # torch.Size([13216, 3])
                       ) -> torch.Tensor:
        lidar_voxel_coords = lidar_voxel_coords.long()
        radar_voxel_coords = radar_voxel_coords.long()

        # [N, 64]
        lidar_pc0_map_vectors = pc0_map[:, lidar_voxel_coords[:, 1], lidar_voxel_coords[:, 2]].T
        radar_pc0_map_vectors = pc0_map[:, radar_voxel_coords[:, 1], radar_voxel_coords[:, 2]].T
        pc0_map_vectors = torch.cat((lidar_pc0_map_vectors, radar_pc0_map_vectors), 0)


        # [N, 64]
        lidar_pc1_map_vectors = pc1_map[:, lidar_voxel_coords[:, 1], lidar_voxel_coords[:, 2]].T
        radar_pc1_map_vectors = pc1_map[:, radar_voxel_coords[:, 1], radar_voxel_coords[:, 2]].T
        pc1_map_vectors = torch.cat((lidar_pc1_map_vectors, radar_pc1_map_vectors), 0)

        # [N, 64]
        lidar_flow_map_vectors = flow_map[:, lidar_voxel_coords[:, 1], lidar_voxel_coords[:, 2]].T
        radar_flow_map_vectors = flow_map[:, radar_voxel_coords[:, 1], radar_voxel_coords[:, 2]].T
        flow_map_vectors = torch.cat((lidar_flow_map_vectors, radar_flow_map_vectors), 0)

        pc_map = self.linear_layer(torch.cat([pc0_map_vectors,pc1_map_vectors], dim=1))

        dynamic_score = self.dy_layer(torch.cat([pc_map, flow_map_vectors], dim=1))

        # return dynamic_score.squeeze(1),masked_flow
        return dynamic_score.squeeze(1)
    

    def forward(
            self, pc0_map: torch.Tensor,   # [B 64 512 512]
            pc1_map: torch.Tensor,   # [B 64 512 512]
            flow_map: torch.Tensor,  # [B 64 512 512]
            voxelizer_infos: List[Dict[str, torch.Tensor]],
            radar_voxelizer_infos: List[Dict[str, torch.Tensor]],
            ) -> List[torch.Tensor]:

        score_results = []
        batch_size = len(pc0_map)

        for batch_id in range(batch_size):
            cur_pc0_map = pc0_map[batch_id]  
            cur_pc1_map = pc1_map[batch_id]  
            cur_flow_map = flow_map[batch_id]  
            # LiDAR
            voxel_coords = voxelizer_infos[batch_id]["voxel_coords"] 
            # Radar
            radar_voxel_coords = radar_voxelizer_infos[batch_id]["voxel_coords"] 
            # temp_score,t = self.seg_single(cur_pc0_map, cur_flow_map, voxel_coords, cur_flow, cur_pc0_fea)
            temp_score = self.seg_single(cur_pc0_map, cur_pc1_map, cur_flow_map, \
                                         voxel_coords, radar_voxel_coords)
            score_results.append(temp_score)

        return score_results
    




















