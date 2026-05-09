import torch
import torch.nn as nn
from typing import List, Tuple, Dict
import copy

SPLIT_BATCH_SIZE = 512

class MMHeadDecoder(nn.Module):

    def __init__(self, pseudoimage_channels: int = 64):
        super().__init__()

        self.offset_encoder = nn.Linear(3, 128)

        # FIXME: figure out how to set nheads and num_layers properly
        # ref: https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html
        # https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoderLayer.html
        transform_decoder_layers = nn.TransformerDecoderLayer(d_model=128, nhead=4)
        self.pts_off_transformer = nn.TransformerDecoder(transform_decoder_layers, num_layers=4)
        
        self.decoder = nn.Sequential(
            nn.Linear(pseudoimage_channels*2, 32), nn.GELU(),
            nn.Linear(32, 3))

    def forward_single(self, before_pseudoimage: torch.Tensor,
                       after_pseudoimage: torch.Tensor,
                       point_offsets: torch.Tensor,
                       voxel_coords: torch.Tensor) -> torch.Tensor:
        voxel_coords = voxel_coords.long()
        # assert (voxel_coords[:, 0] == 0).all(), "Z index must be 0"

        # Voxel coords are Z, Y, X, and the pseudoimage is Channel, Y, X
        # I have confirmed via visualization that these coordinates are correct.
        after_voxel_vectors = after_pseudoimage[:, voxel_coords[:, 1],
                                                voxel_coords[:, 2]].T
        before_voxel_vectors = before_pseudoimage[:, voxel_coords[:, 1],
                                                  voxel_coords[:, 2]].T
        
        # [N, 64] [N, 64] -> [N, 128]
        concatenated_vectors = torch.cat([before_voxel_vectors, after_voxel_vectors], dim=1)
        
        # [N, 128] [N, 128] -> [N, 1, 128]
        voxel_feature = concatenated_vectors.unsqueeze(1)
        point_offsets_feature = self.offset_encoder(point_offsets).unsqueeze(1)
        concatenated_feature = torch.zeros_like(voxel_feature)

        for spilt_range in range(0, concatenated_feature.shape[0], SPLIT_BATCH_SIZE):
            concatenated_feature[spilt_range:spilt_range+SPLIT_BATCH_SIZE] = self.pts_off_transformer(
                voxel_feature[spilt_range:spilt_range+SPLIT_BATCH_SIZE],
                point_offsets_feature[spilt_range:spilt_range+SPLIT_BATCH_SIZE]
            )
        
        flow = self.decoder(concatenated_feature.squeeze(1))
        return flow

    def forward(
            self, before_pseudoimages: torch.Tensor,
            after_pseudoimages: torch.Tensor,
            voxelizer_infos: List[Dict[str,
                                       torch.Tensor]]) -> List[torch.Tensor]:

        flow_results = []
        for before_pseudoimage, after_pseudoimage, voxelizer_info in zip(
                before_pseudoimages, after_pseudoimages, voxelizer_infos):
            point_offsets = voxelizer_info["point_offsets"]
            voxel_coords = voxelizer_info["voxel_coords"]
            flow = self.forward_single(before_pseudoimage, after_pseudoimage,
                                       point_offsets, voxel_coords)
            flow_results.append(flow)
        return flow_results
    
class LinearDecoder(nn.Module):

    def __init__(self, pseudoimage_channels: int = 64):
        super().__init__()

        self.offset_encoder = nn.Linear(3, 128)

        self.decoder = nn.Sequential(
            nn.Linear(pseudoimage_channels*4, 32), nn.GELU(),
            nn.Linear(32, 3))

    def forward_single(self, before_pseudoimage: torch.Tensor,
                       after_pseudoimage: torch.Tensor,
                       point_offsets: torch.Tensor,
                       voxel_coords: torch.Tensor) -> torch.Tensor:
        voxel_coords = voxel_coords.long()
        # assert (voxel_coords[:, 0] == 0).all(), "Z index must be 0"

        # Voxel coords are Z, Y, X, and the pseudoimage is Channel, Y, X
        # I have confirmed via visualization that these coordinates are correct.
        after_voxel_vectors = after_pseudoimage[:, voxel_coords[:, 1],
                                                voxel_coords[:, 2]].T
        before_voxel_vectors = before_pseudoimage[:, voxel_coords[:, 1],
                                                  voxel_coords[:, 2]].T
        
        # [N, 64] [N, 64] -> [N, 128]
        concatenated_vectors = torch.cat([before_voxel_vectors, after_voxel_vectors], dim=1)
        
        # [N, 3] -> [N, 128]
        point_offsets_feature = self.offset_encoder(point_offsets)

        flow = self.decoder(torch.cat([concatenated_vectors, point_offsets_feature], dim=1))
        return flow

    def forward(
            self, before_pseudoimages: torch.Tensor,
            after_pseudoimages: torch.Tensor,
            voxelizer_infos: List[Dict[str,
                                       torch.Tensor]]) -> List[torch.Tensor]:

        flow_results = []
        for before_pseudoimage, after_pseudoimage, voxelizer_info in zip(
                before_pseudoimages, after_pseudoimages, voxelizer_infos):
            point_offsets = voxelizer_info["point_offsets"]
            voxel_coords = voxelizer_info["voxel_coords"]
            flow = self.forward_single(before_pseudoimage, after_pseudoimage,
                                       point_offsets, voxel_coords)
            flow_results.append(flow)
        return flow_results

# from https://github.com/weiyithu/PV-RAFT/blob/main/model/update.py
class ConvGRU(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv1d(input_dim+hidden_dim, hidden_dim, 1)
        self.convr = nn.Conv1d(input_dim+hidden_dim, hidden_dim, 1)
        self.convq = nn.Conv1d(input_dim+hidden_dim, hidden_dim, 1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        rh_x = torch.cat([r*h, x], dim=1)
        q = torch.tanh(self.convq(rh_x))

        h = (1 - z) * h + z * q
        return h
    
class ConvGRUDecoder(nn.Module):

    def __init__(self, pseudoimage_channels: int = 64, num_iters: int = 4):
        super().__init__()

        self.pos_embedding = False
        if self.pos_embedding:
            self.before_layer = nn.Linear(128, 64)

        self.offset_encoder = nn.Linear(3, 64)

        # NOTE: voxel feature is hidden input, point offset is input, check paper's Fig. 3
        self.gru = ConvGRU(input_dim=64, hidden_dim=pseudoimage_channels*2)

        self.decoder = nn.Sequential(
            nn.Linear(pseudoimage_channels*3, 32), nn.GELU(),
            nn.Linear(32, 3))
        self.num_iters = num_iters

    def forward_single(self, before_pseudoimage: torch.Tensor,
                       after_pseudoimage: torch.Tensor,
                       point_offsets: torch.Tensor,
                       voxel_coords: torch.Tensor) -> torch.Tensor:
        voxel_coords = voxel_coords.long()
        # assert (voxel_coords[:, 0] == 0).all(), "Z index must be 0"

        # Voxel coords are Z, Y, X, and the pseudoimage is Channel, Y, X
        # I have confirmed via visualization that these coordinates are correct.
        # indices = coors[:, 1] * self.nx + coors[:, 2]

        after_voxel_vectors = after_pseudoimage[:, voxel_coords[:, 1],
                                                voxel_coords[:, 2]].T
        before_voxel_vectors = before_pseudoimage[:, voxel_coords[:, 1],
                                                  voxel_coords[:, 2]].T
                                                  
        
        # [N, 64] [N, 64] -> [N, 128]
        if self.pos_embedding:
            before_voxel_vectors = self.before_layer(before_voxel_vectors)
        concatenated_vectors = torch.cat([before_voxel_vectors, after_voxel_vectors], dim=1)
        
        # [N, 3] -> [N, 64]
        point_offsets_feature = self.offset_encoder(point_offsets)
        
        # [N, 128] -> [N, 128, 1]
        concatenated_vectors = concatenated_vectors.unsqueeze(2)

        for itr in range(self.num_iters):
            concatenated_vectors = self.gru(concatenated_vectors, point_offsets_feature.unsqueeze(2))

        flow = self.decoder(torch.cat([concatenated_vectors.squeeze(2), point_offsets_feature], dim=1))
        return flow

    def forward(
            self, before_pseudoimages: torch.Tensor,
            after_pseudoimages: torch.Tensor,
            voxelizer_infos: List[Dict[str,
                                       torch.Tensor]]) -> List[torch.Tensor]:

        flow_results = []
        for before_pseudoimage, after_pseudoimage, voxelizer_info in zip(
                before_pseudoimages, after_pseudoimages, voxelizer_infos):
            point_offsets = voxelizer_info["point_offsets"]
            voxel_coords = voxelizer_info["voxel_coords"]
            flow = self.forward_single(before_pseudoimage, after_pseudoimage,
                                       point_offsets, voxel_coords)
            flow_results.append(flow)
        return flow_results


class old_ConvGRUDecoder(nn.Module):

    def __init__(self, pseudoimage_channels: int = 64, num_iters: int = 4):
        super().__init__()

        self.pos_embedding = False
        if self.pos_embedding:
            self.before_layer = nn.Linear(128, 64)

        self.offset_encoder = nn.Linear(3, 64)

        # NOTE: voxel feature is hidden input, point offset is input, check paper's Fig. 3
        self.gru = ConvGRU(input_dim=64, hidden_dim=pseudoimage_channels*2)

        self.decoder = nn.Sequential(
            nn.Linear(pseudoimage_channels*3, 32), nn.GELU(),
            nn.Linear(32, 3))
        self.num_iters = num_iters

    def forward_single(self, before_pseudoimage: torch.Tensor,
                       after_pseudoimage: torch.Tensor,
                       point_offsets: torch.Tensor,
                       voxel_coords: torch.Tensor) -> torch.Tensor:
        voxel_coords = voxel_coords.long()
        # assert (voxel_coords[:, 0] == 0).all(), "Z index must be 0"

        # Voxel coords are Z, Y, X, and the pseudoimage is Channel, Y, X
        # I have confirmed via visualization that these coordinates are correct.
        after_voxel_vectors = after_pseudoimage[:, voxel_coords[:, 1],
                                                voxel_coords[:, 2]].T
        before_voxel_vectors = before_pseudoimage[:, voxel_coords[:, 1],
                                                  voxel_coords[:, 2]].T
        
        # [N, 64] [N, 64] -> [N, 128]
        if self.pos_embedding:
            before_voxel_vectors = self.before_layer(before_voxel_vectors)
        concatenated_vectors = torch.cat([before_voxel_vectors, after_voxel_vectors], dim=1)
        
        # [N, 3] -> [N, 64]
        point_offsets_feature = self.offset_encoder(point_offsets)
        
        # [N, 128] -> [N, 128, 1]
        concatenated_vectors = concatenated_vectors.unsqueeze(2)

        for itr in range(self.num_iters):
            concatenated_vectors = self.gru(concatenated_vectors, point_offsets_feature.unsqueeze(2))

        flow = self.decoder(torch.cat([concatenated_vectors.squeeze(2), point_offsets_feature], dim=1))
        return flow

    def forward(
            self, before_pseudoimages: torch.Tensor,
            after_pseudoimages: torch.Tensor,
            voxelizer_infos: List[Dict[str,
                                       torch.Tensor]]) -> List[torch.Tensor]:

        flow_results = []
        for before_pseudoimage, after_pseudoimage, voxelizer_info in zip(
                before_pseudoimages, after_pseudoimages, voxelizer_infos):
            point_offsets = voxelizer_info["point_offsets"]
            voxel_coords = voxelizer_info["voxel_coords"]
            flow = self.forward_single(before_pseudoimage, after_pseudoimage,
                                       point_offsets, voxel_coords)
            flow_results.append(flow)
        return flow_results


class CylinderGRUDecoder(nn.Module):

    def __init__(self, pseudoimage_channels: int = 64, num_iters: int = 4):
        super().__init__()

        self.offset_encoder = nn.Linear(3, 64)

        # NOTE: voxel feature is hidden input, point offset is input, check paper's Fig. 3
        self.gru = ConvGRU(input_dim=64, hidden_dim=pseudoimage_channels)

        self.decoder = nn.Sequential(
            nn.Linear(pseudoimage_channels*2, 32), nn.GELU(),
            nn.Linear(32, 3))
        self.num_iters = num_iters

    def forward_single(self, before_cylinder_feat, after_cylinder_feat, cylinder_coords, point_feat):
        cor1 = cylinder_coords[:,0].long()
        cor2 = cylinder_coords[:,1].long()
        cor3 = cylinder_coords[:,2].long()

        before_cylinder_to_point_feat = before_cylinder_feat[:, cor1, cor2, cor3].T
        after_cylinder_to_point_feat = after_cylinder_feat[:, cor1, cor2, cor3].T

        # [N, 32] [N, 32] -> [N, 64]
        concat_cylinder_to_pt_feat = torch.cat([before_cylinder_to_point_feat, after_cylinder_to_point_feat], dim=1)

        # [N, 64] -> [N, 64, 1]
        concat_cylinder_to_pt_feat = concat_cylinder_to_pt_feat.unsqueeze(2)

        # point_feat: [N, 64]
        for itr in range(self.num_iters):
            # torch.Size([13272, 64, 1])
            concat_cylinder_to_pt_feat = self.gru(concat_cylinder_to_pt_feat, point_feat.unsqueeze(2))

        flow = self.decoder(torch.cat([concat_cylinder_to_pt_feat.squeeze(2), point_feat], dim=1))

        return flow

        

    def forward(self, before_sp_tensor, after_sp_tensor, pc0_cylinder_infos, pc0_point_feats, batch_size):
        # cylinder_feats: torch.Size([B, 32, 512, 180, 32])
        before_cylinder_feats = before_sp_tensor.dense()
        after_cylinder_feats = after_sp_tensor.dense()
        flow_results = []
        pc0_fea_slice_start = 0

        for batch_idx in range(batch_size):
            # cylinder_feat :torch.Size([32, 512, 180, 32])
            before_cylinder_feat = before_cylinder_feats[batch_idx, :]
            after_cylinder_feat = after_cylinder_feats[batch_idx, :]
        
            # temp_pc0_grid_ind :torch.Size([13272, 3]) 
            temp_pc0_cylinder_ind = pc0_cylinder_infos[batch_idx]
            # 当前batch切片中点的数量，如 13272
            temp_pc0_pts_num = temp_pc0_cylinder_ind.shape[0] 
            # cylinder3D 步骤MLP提取的逐点特征 torch.Size([13272, 64])
            point_feat = pc0_point_feats[pc0_fea_slice_start : (pc0_fea_slice_start + temp_pc0_pts_num), :]
            pc0_fea_slice_start += temp_pc0_pts_num
            
            flow = self.forward_single(before_cylinder_feat, after_cylinder_feat, temp_pc0_cylinder_ind, point_feat)
            # flow.shape: torch.Size([13272, 3]) 
            flow_results.append(flow)

        return flow_results



class PillarSeg(nn.Module):

    def __init__(self, stat_thres: float = 0.5):
        super().__init__()

        self.flow_encoder = nn.Linear(70, 32)
        self.pc_encoder = nn.Linear(96, 32)

        self.mask_decoder = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, 16), nn.GELU(),
            nn.Linear(16, 1))
        
        self.m = nn.Sigmoid()
        self.stat_thres = stat_thres

        # # NOTE: voxel feature is hidden input, point offset is input, check paper's Fig. 3
        # self.gru = ConvGRU(input_dim=64, hidden_dim=pseudoimage_channels*2)

        

        # self.decoder = nn.Sequential(
        #     nn.Linear(pseudoimage_channels*3, 32), nn.GELU(),
        #     nn.Linear(32, 3))
        # self.num_iters = num_iters


    def seg_single(self, pc0_map: torch.Tensor, # torch.Size([64, 512, 512])
                       flow_map: torch.Tensor,  # torch.Size([64, 512, 512])
                       voxel_coords: torch.Tensor, # torch.Size([13216, 3])
                       pc0_fea: torch.Tensor, # torch.Size([13216, 32])
                       flow: torch.Tensor, # torch.Size([13216, 3])
                       pose_flow: torch.Tensor # torch.Size([13216, 3])
                       ) -> torch.Tensor:
        voxel_coords = voxel_coords.long()
        # assert (voxel_coords[:, 0] == 0).all(), "Z index must be 0"

        # Voxel coords are Z, Y, X, and the pseudoimage is Channel, Y, X
        # I have confirmed via visualization that these coordinates are correct.
        pc0_map_vectors = pc0_map[:, voxel_coords[:, 1], voxel_coords[:, 2]].T
        flow_map_vectors = flow_map[:, voxel_coords[:, 1], voxel_coords[:, 2]].T
        
        # [N, 64] [N, 3] [N, 3] -> [N, 70]
        concat_flow_vectors = torch.cat([flow_map_vectors, flow, pose_flow], dim=1)
        # [N, 64] [N, 32] -> [N, 96]
        concat_pc_vectors = torch.cat([pc0_map_vectors, pc0_fea], dim=1)
        
        # [N, 70] -> [N, 32]
        flow_feature = self.flow_encoder(concat_flow_vectors)
        
        # [N, 96] -> [N, 32]
        pc_feature = self.pc_encoder(concat_pc_vectors)

        # [N, 32] [N, 32] -> [N, 64]
        # final_pc_fea = torch.cat([flow_feature, pc_feature], dim=1)
        # [N, 1] dynamic score
        dynamic_score = self.m(self.mask_decoder(torch.cat([flow_feature, pc_feature], dim=1)))

        # threshold the probabilities to obtain the binary mask output
        # torch.Size([13216])
        mask = (dynamic_score.squeeze(1) > self.stat_thres)

        return mask, dynamic_score.squeeze(1)

    def forward(
            self, pc0_map: torch.Tensor,   # [B 128 512 512]
            flow_map: torch.Tensor,  # [B 64 512 512]
            voxelizer_infos: List[Dict[str,
                                       torch.Tensor]],
            pc0_point_fea: List[torch.Tensor],
            flows: List[torch.Tensor],
            pose_flows: List[torch.Tensor],
            radar_pose_flows: List[torch.Tensor]) -> List[torch.Tensor]:

        seg_results = []
        score_results = []
        batch_size = len(pc0_point_fea)

        for batch_id in range(batch_size):
            cur_pc0_map = pc0_map[batch_id]  
            cur_flow_map = flow_map[batch_id]  
            voxel_coords = voxelizer_infos[batch_id]["voxel_coords"] 
            cur_pc0_fea = pc0_point_fea[batch_id] 
            cur_flow = flows[batch_id] 
            cur_pose_flow = torch.cat((pose_flows[batch_id], radar_pose_flows[batch_id]), dim=0) # torch.Size([14544, 3])
            # select valid pose flow
            cur_valid_pt_idxes = voxelizer_infos[batch_id]["point_idxes"]
            cur_pose_flow = cur_pose_flow[cur_valid_pt_idxes,:] 

            temp_seg, temp_score = self.seg_single(cur_pc0_map, cur_flow_map, voxel_coords,\
                                        cur_pc0_fea, cur_flow ,cur_pose_flow)
            
            seg_results.append(temp_seg)
            score_results.append(temp_score)

        return seg_results, score_results



class Modal_Seg(nn.Module):
    def __init__(self, stat_thres: float = 0.5):
        super().__init__()

        self.flow_encoder = nn.Linear(3, 32)
        self.flow_map_encoder = nn.Linear(64, 32)
        self.pc_encoder = nn.Linear(64, 32)

        self.mask_decoder = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, 16), nn.GELU(),
            nn.Linear(16, 1))
        
        self.m = nn.Sigmoid()
        self.stat_thres = stat_thres

        # # NOTE: voxel feature is hidden input, point offset is input, check paper's Fig. 3
        # self.gru = ConvGRU(input_dim=64, hidden_dim=pseudoimage_channels*2)

        

        # self.decoder = nn.Sequential(
        #     nn.Linear(pseudoimage_channels*3, 32), nn.GELU(),
        #     nn.Linear(32, 3))
        # self.num_iters = num_iters

    # self.seg_single(cur_pc0_map, cur_flow_map, voxel_coords, cur_flow )
    def seg_single(self, pc0_map: torch.Tensor, # torch.Size([64, 512, 512])
                       flow_map: torch.Tensor,  # torch.Size([64, 512, 512])
                       voxel_coords: torch.Tensor, # torch.Size([13216, 3])
                       flow: torch.Tensor, # torch.Size([13216, 3])
                       pc0_fea: torch.Tensor
                       ) -> torch.Tensor:
        voxel_coords = voxel_coords.long()
        # assert (voxel_coords[:, 0] == 0).all(), "Z index must be 0"

        # Voxel coords are Z, Y, X, and the pseudoimage is Channel, Y, X
        # I have confirmed via visualization that these coordinates are correct.
        # [N, 64]
        pc0_map_vectors = pc0_map[:, voxel_coords[:, 1], voxel_coords[:, 2]].T
        # [N, 64]
        flow_map_vectors = flow_map[:, voxel_coords[:, 1], voxel_coords[:, 2]].T
        # [N, 3] -> [N, 32]
        flow_fea = self.flow_encoder(flow)
        # [N, 64] -> [N, 32]
        flow_map_fea = self.flow_map_encoder(flow_map_vectors)
        # [N, 64] -> [N, 32]
        pc_map_fea = self.pc_encoder(pc0_map_vectors)
        # [N, 96]  [N, 128] 
        all_fea = torch.cat([flow_fea, flow_map_fea, pc_map_fea, pc0_fea], dim=1)


        # [N, 32] [N, 32] -> [N, 64]
        # final_pc_fea = torch.cat([flow_feature, pc_feature], dim=1)
        # [N, 1] dynamic score
        dynamic_score = self.m(self.mask_decoder(all_fea))

        # threshold the probabilities to obtain the binary mask output
        # torch.Size([13216])
        mask = (dynamic_score.squeeze(1) > self.stat_thres)

        return mask, dynamic_score.squeeze(1)

    def forward(
            self, pc0_map: torch.Tensor,   # [B 128 512 512]
            flow_map: torch.Tensor,  # [B 64 512 512]
            voxelizer_infos: List[Dict[str,
                                       torch.Tensor]],
            pc0_point_fea: List[torch.Tensor],                        
            flows: List[torch.Tensor]) -> List[torch.Tensor]:

        seg_results = []
        score_results = []
        batch_size = len(flows)

        for batch_id in range(batch_size):
            cur_pc0_map = pc0_map[batch_id]  
            cur_flow_map = flow_map[batch_id]  
            voxel_coords = voxelizer_infos[batch_id]["voxel_coords"] 
            cur_flow = flows[batch_id] 
            cur_pc0_fea = pc0_point_fea[batch_id]
            temp_seg, temp_score = self.seg_single(cur_pc0_map, cur_flow_map, voxel_coords, cur_flow, cur_pc0_fea)
            
            seg_results.append(temp_seg)
            score_results.append(temp_score)

        return seg_results, score_results



class Cascade_Modal_Seg(nn.Module):

    def __init__(self, stat_thres: float = 0.5):
        super().__init__()

        self.flow_encoder = nn.Linear(3, 32)
        self.flow_map_encoder = nn.Linear(64, 32)
        self.pc_encoder = nn.Linear(64, 32)

        self.dy_layer1 =nn.Sequential(
                    nn.Linear(64, 32),
                    nn.BatchNorm1d(32, eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True))
        
        # dy_layer2 = []
        # dy_layer2.append(nn.Sequential(
        #             nn.Linear(96, 32, bias=False),
        #             nn.BatchNorm1d(32, eps=1e-3, momentum=0.01),
        #             nn.ReLU(inplace=True)))
        # dy_layer2.append(nn.Sequential(
        #             nn.Linear(32, 16, bias=False),
        #             nn.BatchNorm1d(16, eps=1e-3, momentum=0.01),
        #             nn.ReLU(inplace=True)))
        # dy_layer2.append(nn.Sequential(
        #             nn.Linear(16, 1, bias=False),
        #             nn.BatchNorm1d(1, eps=1e-3, momentum=0.01),
        #             nn.ReLU(inplace=True)),
        #             nn.Sigmoid())
        # self.dy_layer2 = nn.ModuleList(dy_layer2)  

        self.dy_layer2 =nn.Sequential(
                    nn.Linear(96, 32), nn.GELU(),
                    nn.Linear(32, 16), nn.GELU(),
                    nn.Linear(16, 1), nn.Sigmoid()
                    )
        

        # self.mask_decoder = nn.Sequential(
        #     nn.Linear(128, 64), nn.GELU(),
        #     nn.Linear(64, 32), nn.GELU(),
        #     nn.Linear(32, 16), nn.GELU(),
        #     nn.Linear(16, 1))
        # self.m = nn.Sigmoid()
        self.stat_thres = stat_thres

        # # NOTE: voxel feature is hidden input, point offset is input, check paper's Fig. 3
        # self.gru = ConvGRU(input_dim=64, hidden_dim=pseudoimage_channels*2)

        

        # self.decoder = nn.Sequential(
        #     nn.Linear(pseudoimage_channels*3, 32), nn.GELU(),
        #     nn.Linear(32, 3))
        # self.num_iters = num_iters

    # self.seg_single(cur_pc0_map, cur_flow_map, voxel_coords, cur_flow )
    def seg_single(self, pc0_map: torch.Tensor, # torch.Size([64, 512, 512])
                       flow_map: torch.Tensor,  # torch.Size([64, 512, 512])
                       voxel_coords: torch.Tensor, # torch.Size([13216, 3])
                       flow: torch.Tensor, # torch.Size([13216, 3])
                       pc0_fea: torch.Tensor
                       ) -> torch.Tensor:
        voxel_coords = voxel_coords.long()
        # assert (voxel_coords[:, 0] == 0).all(), "Z index must be 0"

        # Voxel coords are Z, Y, X, and the pseudoimage is Channel, Y, X
        # I have confirmed via visualization that these coordinates are correct.
        # [N, 64]
        pc0_map_vectors = pc0_map[:, voxel_coords[:, 1], voxel_coords[:, 2]].T
        # [N, 64]
        flow_map_vectors = flow_map[:, voxel_coords[:, 1], voxel_coords[:, 2]].T
        # [N, 3] -> [N, 32]
        flow_fea = self.flow_encoder(flow)
        # [N, 64] -> [N, 32]
        flow_map_fea = self.flow_map_encoder(flow_map_vectors)
        # [N, 64] -> [N, 32]
        pc_map_fea = self.pc_encoder(pc0_map_vectors)


        # Version 1
        # [N, 96]  [N, 128] 
        # all_fea = torch.cat([flow_fea, flow_map_fea, pc_map_fea, pc0_fea], dim=1)
        # [N, 32] [N, 32] -> [N, 64]
        # final_pc_fea = torch.cat([flow_feature, pc_feature], dim=1)
        # [N, 1] dynamic score
        # dynamic_score = self.m(self.mask_decoder(all_fea))

        # Version 2
        # [N, 64]
        fea_layer1 = self.dy_layer1(torch.cat([flow_fea,  pc_map_fea], dim=1))
        dynamic_score =self.dy_layer2(torch.cat([fea_layer1,  flow_map_fea, pc0_fea], dim=1))


       

        # threshold the probabilities to obtain the binary mask output
        # torch.Size([13216])
        mask = (dynamic_score.squeeze(1) > self.stat_thres)

        return mask, dynamic_score.squeeze(1)

    def forward(
            self, pc0_map: torch.Tensor,   # [B 128 512 512]
            flow_map: torch.Tensor,  # [B 64 512 512]
            voxelizer_infos: List[Dict[str,
                                       torch.Tensor]],
            pc0_point_fea: List[torch.Tensor],                        
            flows: List[torch.Tensor]) -> List[torch.Tensor]:

        seg_results = []
        score_results = []
        batch_size = len(flows)

        for batch_id in range(batch_size):
            cur_pc0_map = pc0_map[batch_id]  
            cur_flow_map = flow_map[batch_id]  
            voxel_coords = voxelizer_infos[batch_id]["voxel_coords"] 
            cur_flow = flows[batch_id] 
            cur_pc0_fea = pc0_point_fea[batch_id]
            temp_seg, temp_score = self.seg_single(cur_pc0_map, cur_flow_map, voxel_coords, cur_flow, cur_pc0_fea)
            
            seg_results.append(temp_seg)
            score_results.append(temp_score)

        return seg_results, score_results




class Cascade_Modal_Seg_v2(nn.Module):

    def __init__(self, stat_thres: float = 0.5):
        super().__init__()

        self.flow_encoder = nn.Linear(3, 32)
        self.flow_map_encoder = nn.Linear(64, 32)
        self.pc_encoder = nn.Linear(64, 32)

        self.dy_layer1 =nn.Sequential(
                    nn.Linear(64, 32),
                    nn.BatchNorm1d(32, eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True))
        
        # dy_layer2 = []
        # dy_layer2.append(nn.Sequential(
        #             nn.Linear(96, 32, bias=False),
        #             nn.BatchNorm1d(32, eps=1e-3, momentum=0.01),
        #             nn.ReLU(inplace=True)))
        # dy_layer2.append(nn.Sequential(
        #             nn.Linear(32, 16, bias=False),
        #             nn.BatchNorm1d(16, eps=1e-3, momentum=0.01),
        #             nn.ReLU(inplace=True)))
        # dy_layer2.append(nn.Sequential(
        #             nn.Linear(16, 1, bias=False),
        #             nn.BatchNorm1d(1, eps=1e-3, momentum=0.01),
        #             nn.ReLU(inplace=True)),
        #             nn.Sigmoid())
        # self.dy_layer2 = nn.ModuleList(dy_layer2)  

        self.dy_layer2 =nn.Sequential(
                    nn.Linear(96, 32), nn.GELU(),
                    nn.Linear(32, 16), nn.GELU(),
                    nn.Linear(16, 1), nn.Sigmoid()
                    )
        

        # self.mask_decoder = nn.Sequential(
        #     nn.Linear(128, 64), nn.GELU(),
        #     nn.Linear(64, 32), nn.GELU(),
        #     nn.Linear(32, 16), nn.GELU(),
        #     nn.Linear(16, 1))
        # self.m = nn.Sigmoid()
        self.stat_thres = stat_thres

        # # NOTE: voxel feature is hidden input, point offset is input, check paper's Fig. 3
        # self.gru = ConvGRU(input_dim=64, hidden_dim=pseudoimage_channels*2)

        

        # self.decoder = nn.Sequential(
        #     nn.Linear(pseudoimage_channels*3, 32), nn.GELU(),
        #     nn.Linear(32, 3))
        # self.num_iters = num_iters

    # self.seg_single(cur_pc0_map, cur_flow_map, voxel_coords, cur_flow )
    def seg_single(self, pc0_map: torch.Tensor, # torch.Size([64, 512, 512])
                       flow_map: torch.Tensor,  # torch.Size([64, 512, 512])
                       voxel_coords: torch.Tensor, # torch.Size([13216, 3])
                       flow: torch.Tensor, # torch.Size([13216, 3])
                       pc0_fea: torch.Tensor
                       ) -> torch.Tensor:
        voxel_coords = voxel_coords.long()
        # assert (voxel_coords[:, 0] == 0).all(), "Z index must be 0"

        # Voxel coords are Z, Y, X, and the pseudoimage is Channel, Y, X
        # I have confirmed via visualization that these coordinates are correct.
        # [N, 64]
        pc0_map_vectors = pc0_map[:, voxel_coords[:, 1], voxel_coords[:, 2]].T
        # [N, 64]
        flow_map_vectors = flow_map[:, voxel_coords[:, 1], voxel_coords[:, 2]].T

        copy_pc0_fea = copy.deepcopy(pc0_fea)
        copy_flow = copy.deepcopy(flow)
        copy_flow_map_vectors = copy.deepcopy(flow_map_vectors)
        copy_pc0_map_vectors = copy.deepcopy(pc0_map_vectors)

        copy_flow.requires_grad = True
        copy_flow_map_vectors.requires_grad = True
        copy_pc0_map_vectors.requires_grad = True

        # [N, 3] -> [N, 32]
        flow_fea = self.flow_encoder(copy_flow)
        # [N, 64] -> [N, 32]
        flow_map_fea = self.flow_map_encoder(copy_flow_map_vectors)
        # [N, 64] -> [N, 32]
        pc_map_fea = self.pc_encoder(copy_pc0_map_vectors)


        # Version 1
        # [N, 96]  [N, 128] 
        # all_fea = torch.cat([flow_fea, flow_map_fea, pc_map_fea, pc0_fea], dim=1)
        # [N, 32] [N, 32] -> [N, 64]
        # final_pc_fea = torch.cat([flow_feature, pc_feature], dim=1)
        # [N, 1] dynamic score
        # dynamic_score = self.m(self.mask_decoder(all_fea))

        # Version 2
        # [N, 64]

        fea_layer1 = self.dy_layer1(torch.cat([flow_fea,  pc_map_fea], dim=1))
        dynamic_score =self.dy_layer2(torch.cat([fea_layer1,  flow_map_fea, copy_pc0_fea], dim=1))
        # mask = (dynamic_score.squeeze(1) > self.stat_thres)

        

        x = dynamic_score.expand(-1, 3)        
        masked_flow = copy_flow.masked_fill(x < self.stat_thres, 0)


       

        # threshold the probabilities to obtain the binary mask output
        # torch.Size([13216])
        # mask = (dynamic_score.squeeze(1) > self.stat_thres)

        return dynamic_score.squeeze(1),masked_flow
    def forward(
            self, pc0_map: torch.Tensor,   # [B 128 512 512]
            flow_map: torch.Tensor,  # [B 64 512 512]
            voxelizer_infos: List[Dict[str,
                                       torch.Tensor]],
            pc0_point_fea: List[torch.Tensor],                        
            flows: List[torch.Tensor]) -> List[torch.Tensor]:

        score_results = []
        batch_size = len(flows)
        flow = []

        for batch_id in range(batch_size):
            cur_pc0_map = pc0_map[batch_id]  
            cur_flow_map = flow_map[batch_id]  
            voxel_coords = voxelizer_infos[batch_id]["voxel_coords"] 
            cur_flow = flows[batch_id] 
            cur_pc0_fea = pc0_point_fea[batch_id]
            temp_score,t = self.seg_single(cur_pc0_map, cur_flow_map, voxel_coords, cur_flow, cur_pc0_fea)

            score_results.append(temp_score)
            flow.append(t)

        return score_results,flow







    
class Cascade_Modal_Seg_v3(nn.Module):
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
                       voxel_coords: torch.Tensor, # torch.Size([13216, 3])
                       flow: torch.Tensor, # torch.Size([13216, 3])
                       pc0_fea: torch.Tensor
                       ) -> torch.Tensor:
        voxel_coords = voxel_coords.long()
        # assert (voxel_coords[:, 0] == 0).all(), "Z index must be 0"

        # Voxel coords are Z, Y, X, and the pseudoimage is Channel, Y, X
        # I have confirmed via visualization that these coordinates are correct.
        # [N, 64]
        pc0_map_vectors = pc0_map[:, voxel_coords[:, 1], voxel_coords[:, 2]].T
        # [N, 64]
        flow_map_vectors = flow_map[:, voxel_coords[:, 1], voxel_coords[:, 2]].T


        pc0_fea.requires_grad = True
        flow.requires_grad = True
        flow_map_vectors.requires_grad = True
        pc0_map_vectors.requires_grad = True

        # [N, 3] -> [N, 32]
        flow_fea = self.flow_encoder(flow)
        # [N, 64] -> [N, 32]
        flow_map_fea = self.flow_map_encoder(flow_map_vectors)
        # [N, 64] -> [N, 32]
        pc_map_fea = self.pc_encoder(pc0_map_vectors)

        fea_layer1 = self.dy_layer1(torch.cat([flow_fea,  pc_map_fea], dim=1))
        dynamic_score =self.dy_layer2(torch.cat([fea_layer1,  flow_map_fea, pc0_fea], dim=1))

        x = dynamic_score.expand(-1, 3)        
        masked_flow = flow.masked_fill(x < self.stat_thres, 0)

        return dynamic_score.squeeze(1),masked_flow
    

    def forward(
            self, pc0_map: torch.Tensor,   # [B 128 512 512]
            flow_map: torch.Tensor,  # [B 64 512 512]
            voxelizer_infos: List[Dict[str,
                                       torch.Tensor]],
            pc0_point_fea: List[torch.Tensor],                        
            flows: List[torch.Tensor]) -> List[torch.Tensor]:

        score_results = []
        batch_size = len(flows)
        flow = []

        for batch_id in range(batch_size):
            cur_pc0_map = pc0_map[batch_id]  
            cur_flow_map = flow_map[batch_id]  
            voxel_coords = voxelizer_infos[batch_id]["voxel_coords"] 
            cur_flow = flows[batch_id] 
            cur_pc0_fea = pc0_point_fea[batch_id]
            temp_score,t = self.seg_single(cur_pc0_map, cur_flow_map, voxel_coords, cur_flow, cur_pc0_fea)

            score_results.append(temp_score)
            flow.append(t)

        return score_results,flow
    




class Cascade_Modal_Seg_v4(nn.Module):
    def __init__(self, stat_thres: float = 0.5):
        super().__init__()

        self.flow_encoder = nn.Linear(3, 32)
        self.flow_map_encoder = nn.Linear(64, 32)
        self.pc_encoder = nn.Linear(32, 32)  # input pc0_map size = 32

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
                       voxel_coords: torch.Tensor, # torch.Size([13216, 3])
                       flow: torch.Tensor, # torch.Size([13216, 3])
                       pc0_fea: torch.Tensor
                       ) -> torch.Tensor:
        voxel_coords = voxel_coords.long()
        # assert (voxel_coords[:, 0] == 0).all(), "Z index must be 0"

        # Voxel coords are Z, Y, X, and the pseudoimage is Channel, Y, X
        # I have confirmed via visualization that these coordinates are correct.
        # [N, 64]
        pc0_map_vectors = pc0_map[:, voxel_coords[:, 1], voxel_coords[:, 2]].T
        # [N, 64]
        flow_map_vectors = flow_map[:, voxel_coords[:, 1], voxel_coords[:, 2]].T


        pc0_fea.requires_grad = True
        flow.requires_grad = True
        flow_map_vectors.requires_grad = True
        pc0_map_vectors.requires_grad = True

        # [N, 3] -> [N, 32]
        flow_fea = self.flow_encoder(flow)
        # [N, 64] -> [N, 32]
        flow_map_fea = self.flow_map_encoder(flow_map_vectors)
        # [N, 64] -> [N, 32]
        pc_map_fea = self.pc_encoder(pc0_map_vectors)

        fea_layer1 = self.dy_layer1(torch.cat([flow_fea,  pc_map_fea], dim=1))
        dynamic_score =self.dy_layer2(torch.cat([fea_layer1,  flow_map_fea, pc0_fea], dim=1))

        x = dynamic_score.expand(-1, 3)        
        masked_flow = flow.masked_fill(x < self.stat_thres, 0)

        return dynamic_score.squeeze(1),masked_flow
    

    def forward(
            self, pc0_map: torch.Tensor,   # [B 128 512 512]
            flow_map: torch.Tensor,  # [B 64 512 512]
            voxelizer_infos: List[Dict[str,
                                       torch.Tensor]],
            pc0_point_fea: List[torch.Tensor],                        
            flows: List[torch.Tensor]) -> List[torch.Tensor]:

        score_results = []
        batch_size = len(flows)
        flow = []

        for batch_id in range(batch_size):
            cur_pc0_map = pc0_map[batch_id]  
            cur_flow_map = flow_map[batch_id]  
            voxel_coords = voxelizer_infos[batch_id]["voxel_coords"] 
            cur_flow = flows[batch_id] 
            cur_pc0_fea = pc0_point_fea[batch_id]
            temp_score,t = self.seg_single(cur_pc0_map, cur_flow_map, voxel_coords, cur_flow, cur_pc0_fea)

            score_results.append(temp_score)
            flow.append(t)

        return score_results,flow




class Cascade_Modal_Seg_Self(nn.Module):

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
                       voxel_coords: torch.Tensor, # torch.Size([13216, 3])
                       flow: torch.Tensor, # torch.Size([13216, 3])
                       pc0_fea: torch.Tensor
                       ) -> torch.Tensor:
        voxel_coords = voxel_coords.long()
        # assert (voxel_coords[:, 0] == 0).all(), "Z index must be 0"


        # Voxel coords are Z, Y, X, and the pseudoimage is Channel, Y, X
        # I have confirmed via visualization that these coordinates are correct.
        # [N, 64]
        pc0_map_vectors = pc0_map[:, voxel_coords[:, 1], voxel_coords[:, 2]].T
        # [N, 64]
        flow_map_vectors = flow_map[:, voxel_coords[:, 1], voxel_coords[:, 2]].T


        # pc0_fea.requires_grad = True
        # flow.requires_grad = True
        # flow_map_vectors.requires_grad = True
        # pc0_map_vectors.requires_grad = True

        # [N, 3] -> [N, 32]
        flow_fea = self.flow_encoder(flow)
        # [N, 64] -> [N, 32]
        flow_map_fea = self.flow_map_encoder(flow_map_vectors)
        # [N, 64] -> [N, 32]
        pc_map_fea = self.pc_encoder(pc0_map_vectors)

        fea_layer1 = self.dy_layer1(torch.cat([flow_fea,  pc_map_fea], dim=1))
        dynamic_score =self.dy_layer2(torch.cat([fea_layer1,  flow_map_fea, pc0_fea], dim=1))

        x = dynamic_score.expand(-1, 3)        
        masked_flow = flow.masked_fill(x < self.stat_thres, 0)

        return dynamic_score.squeeze(1),masked_flow
    

    def forward(
            self, pc0_map: torch.Tensor,   # [B 64 512 512]
            flow_map: torch.Tensor,  # [B 64 512 512]
            voxelizer_infos: List[Dict[str,
                                       torch.Tensor]],
            pc0_point_fea: List[torch.Tensor],     # [B N 32]                   
            flows: List[torch.Tensor]) -> List[torch.Tensor]:  # [B N 3]      

        score_results = []
        batch_size = len(flows)
        flow = []

        for batch_id in range(batch_size):
            cur_pc0_map = pc0_map[batch_id]  
            cur_flow_map = flow_map[batch_id]  
            voxel_coords = voxelizer_infos[batch_id]["voxel_coords"] 
            cur_flow = flows[batch_id] 
            cur_pc0_fea = pc0_point_fea[batch_id]
            temp_score,t = self.seg_single(cur_pc0_map, cur_flow_map, voxel_coords, cur_flow, cur_pc0_fea)

            score_results.append(temp_score)
            flow.append(t)

        return score_results,flow





class Cascade_Modal_onlySeg_Self(nn.Module):

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
                       voxel_coords: torch.Tensor, # torch.Size([13216, 3])
                       flow: torch.Tensor, # torch.Size([13216, 3])
                       pc0_fea: torch.Tensor
                       ) -> torch.Tensor:
        voxel_coords = voxel_coords.long()
        # assert (voxel_coords[:, 0] == 0).all(), "Z index must be 0"


        # Voxel coords are Z, Y, X, and the pseudoimage is Channel, Y, X
        # I have confirmed via visualization that these coordinates are correct.
        # [N, 64]
        pc0_map_vectors = pc0_map[:, voxel_coords[:, 1], voxel_coords[:, 2]].T
        # [N, 64]
        flow_map_vectors = flow_map[:, voxel_coords[:, 1], voxel_coords[:, 2]].T


        # pc0_fea.requires_grad = True
        # flow.requires_grad = True
        # flow_map_vectors.requires_grad = True
        # pc0_map_vectors.requires_grad = True

        # [N, 3] -> [N, 32]
        flow_fea = self.flow_encoder(flow)
        # [N, 64] -> [N, 32]
        flow_map_fea = self.flow_map_encoder(flow_map_vectors)
        # [N, 64] -> [N, 32]
        pc_map_fea = self.pc_encoder(pc0_map_vectors)

        fea_layer1 = self.dy_layer1(torch.cat([flow_fea,  pc_map_fea], dim=1))
        dynamic_score =self.dy_layer2(torch.cat([fea_layer1,  flow_map_fea, pc0_fea], dim=1))

        # x = dynamic_score.expand(-1, 3)        
        # masked_flow = flow.masked_fill(x < self.stat_thres, 0)

        # return dynamic_score.squeeze(1),masked_flow
        return dynamic_score.squeeze(1)
    

    def forward(
            self, pc0_map: torch.Tensor,   # [B 64 512 512]
            flow_map: torch.Tensor,  # [B 64 512 512]
            voxelizer_infos: List[Dict[str,
                                       torch.Tensor]],
            pc0_point_fea: List[torch.Tensor],     # [B N 32]                   
            flows: List[torch.Tensor]) -> List[torch.Tensor]:  # [B N 3]      

        score_results = []
        batch_size = len(flows)
        # flow = []

        for batch_id in range(batch_size):
            cur_pc0_map = pc0_map[batch_id]  
            cur_flow_map = flow_map[batch_id]  
            voxel_coords = voxelizer_infos[batch_id]["voxel_coords"] 
            cur_flow = flows[batch_id] 
            cur_pc0_fea = pc0_point_fea[batch_id]
            temp_score = self.seg_single(cur_pc0_map, cur_flow_map, voxel_coords, cur_flow, cur_pc0_fea)

            score_results.append(temp_score)
            # flow.append(t)

        # return score_results,flow
        return score_results








class Cascade_Seg_score(nn.Module):
    def __init__(self, stat_thres: float = 0.5):
        super().__init__()

        self.pt_offset_encoder = nn.Linear(3, 32)
        self.pseudo_encoder = nn.Linear(64, 32)
        self.flow_map_encoder = nn.Linear(64, 32)

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


    def seg_single(self, pseudo_map: torch.Tensor, # torch.Size([64, 512, 512])
                       flow_map: torch.Tensor,  # torch.Size([64, 512, 512])
                       voxel_coords: torch.Tensor, # torch.Size([13216, 3])
                       pc0_offset: torch.Tensor  # torch.Size([13216, 3])
                       ) -> torch.Tensor:
        voxel_coords = voxel_coords.long()
        # assert (voxel_coords[:, 0] == 0).all(), "Z index must be 0"

        # Voxel coords are Z, Y, X, and the pseudoimage is Channel, Y, X
        # I have confirmed via visualization that these coordinates are correct.
        # [N, 64]
        pseudo_map_vectors = pseudo_map[:, voxel_coords[:, 1], voxel_coords[:, 2]].T
        # [N, 64]
        flow_map_vectors = flow_map[:, voxel_coords[:, 1], voxel_coords[:, 2]].T


        # pc0_fea.requires_grad = True
        # flow.requires_grad = True
        # flow_map_vectors.requires_grad = True
        # pc0_map_vectors.requires_grad = True

        # [N, 64] -> [N, 32]
        flow_map_fea = self.flow_map_encoder(flow_map_vectors)
        # [N, 64] -> [N, 32]
        pseudo_map_fea = self.pseudo_encoder(pseudo_map_vectors)
        offset_fea = self.pt_offset_encoder(pc0_offset)

        fea_layer1 = self.dy_layer1(torch.cat([flow_map_fea,  pseudo_map_fea], dim=1))
        dynamic_score =self.dy_layer2(torch.cat([fea_layer1,  offset_fea], dim=1))

        return dynamic_score.squeeze(1)
    

    def forward(
            self, pseudo_map: torch.Tensor,   # [B 128 512 512]
            flow_map: torch.Tensor,  # [B 64 512 512]
            voxelizer_infos: List[Dict[str,
                                       torch.Tensor]]
            ) -> List[torch.Tensor]:

        score_results = []
        batch_size = pseudo_map.shape[0]

        for batch_id in range(batch_size):
            cur_pseudo_map = pseudo_map[batch_id]  
            cur_flow_map = flow_map[batch_id]  
            voxel_coords = voxelizer_infos[batch_id]["voxel_coords"] 
            point_offsets = voxelizer_infos[batch_id]["point_offsets"]

            temp_score = self.seg_single(cur_pseudo_map, cur_flow_map, voxel_coords, point_offsets)

            score_results.append(temp_score)


        return score_results
