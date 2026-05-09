import torch
import torch.nn as nn

from .make_voxels import  DynamicVoxelizer  # HardVoxelizer,
from .process_voxels import Fusion_DynamicPillarFeatureNet, my_PillarFeatureNet # DynamicPillarFeatureNet # PillarFeatureNet,  
from .scatter import PointPillarsScatter, my_PointPillarsScatter
from .bev_shift import shift_bev_grids, return_tensor_index
# import torch_scatter
from mmcv.ops import DynamicScatter


INDEX_SHIFT = [[0,0], [-1,0], [1,0], [0,1], [-1,1], [1,1], [0,-1], [-1,-1], [1,-1]]
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



class my_DynamicEmbedder(nn.Module):
    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 in_channel: int, feat_channel: int) -> None:
        super().__init__()
        self.pseudo_image_dims = pseudo_image_dims
        
        self.voxelizer = DynamicVoxelizer(voxel_size=voxel_size,
                                          point_cloud_range=point_cloud_range)
        
        self.feature_net = my_PillarFeatureNet(
            in_channels = in_channel,
            feat_channels = feat_channel, 
            point_cloud_range=point_cloud_range,
            voxel_size = voxel_size,
            mode='avg') # 'avg'
        
        self.scatter = my_PointPillarsScatter(in_channels=feat_channel, output_shape=pseudo_image_dims)

    def forward(self, points: torch.Tensor, points_in_fea: torch.Tensor) -> torch.Tensor:
        
        voxel_info_list = self.voxelizer(points)

        pseudoimage_lst = []
        pt_fea_lst = []
        out_bev_feats_lst = []
        out_bev_coors_lst = []

        for batch_id, voxel_info_dict in enumerate(voxel_info_list):
            points = voxel_info_dict['points']
            # voxel_size: [0.1, 0.4, 6]
            # coord: [ 0, 113, 322]
            coordinates = voxel_info_dict['voxel_coords']
            # my EDIT
            valid_point_id = voxel_info_dict['point_idxes']
            temp_pts_fea_batch = points_in_fea[batch_id]
            temp_pts_fea = temp_pts_fea_batch[valid_point_id,:]
           
            # [ 0, 113, 325],
            voxel_feats, voxel_coors, point_feats = self.feature_net(points, temp_pts_fea, coordinates)
            # voxel_feats : torch.Size([NUM_non_empty_voxel, 32])
            # voxel_coors : torch.Size([NUM_non_empty_voxel, 3])   Example [ 0,  48, 331]
            # point_feats : torch.Size([ N_points , 32])

            pt_fea_lst.append(point_feats)
            
            bev_coors = voxel_coors[:,1:]
            pseudoimage = self.scatter(voxel_feats, voxel_coors)
            pseudoimage_lst.append(pseudoimage)

            out_bev_feats_lst.append(voxel_feats)
            out_bev_coors_lst.append(bev_coors)
            
        # Concatenate the pseudoimages along the batch dimension
        return torch.cat(pseudoimage_lst, dim=0), voxel_info_list, pt_fea_lst, out_bev_feats_lst, out_bev_coors_lst
    



class try_DynamicEmbedder(nn.Module):
    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 in_channel: int, feat_channel: int) -> None:
        super().__init__()
        self.pseudo_image_dims = pseudo_image_dims
        
        # self.voxelizer = DynamicVoxelizer(voxel_size=voxel_size,
        #                                   point_cloud_range=point_cloud_range)
        
        self.feature_net = my_PillarFeatureNet(
            in_channels = in_channel,
            feat_channels = feat_channel, 
            point_cloud_range=point_cloud_range,
            voxel_size = voxel_size,
            mode='avg') # 'avg'
        
        self.scatter = my_PointPillarsScatter(in_channels=feat_channel, output_shape=pseudo_image_dims)

    def forward(self, voxel_info_list) -> torch.Tensor:
    # def forward(self, points_in_fea, voxel_info_list) -> torch.Tensor:
        
        # voxel_info_list = self.voxelizer(points)

        pseudoimage_lst = []
        pt_fea_lst = []
        out_bev_feats_lst = []
        out_bev_coors_lst = []

        for batch_id, voxel_info_dict in enumerate(voxel_info_list):
            points = voxel_info_dict['points']
            # voxel_size: [0.1, 0.4, 6]
            # coord: [ 0, 113, 322]
            coordinates = voxel_info_dict['voxel_coords']
            # my EDIT
            # valid_point_id = voxel_info_dict['point_idxes']
            # temp_pts_fea_batch = points_in_fea[batch_id]
            # temp_pts_fea = temp_pts_fea_batch[valid_point_id,:]

            # temp_pts_fea = points_in_fea[batch_id]
            temp_pts_fea = points
           
            # [ 0, 113, 325],
            voxel_feats, voxel_coors, point_feats = self.feature_net(points, temp_pts_fea, coordinates)
            # voxel_feats : torch.Size([NUM_non_empty_voxel, 32])
            # voxel_coors : torch.Size([NUM_non_empty_voxel, 3])   Example [ 0,  48, 331]
            # point_feats : torch.Size([ N_points , 32])

            pt_fea_lst.append(point_feats)
            
            bev_coors = voxel_coors[:,1:]
            pseudoimage = self.scatter(voxel_feats, voxel_coors)
            pseudoimage_lst.append(pseudoimage)

            out_bev_feats_lst.append(voxel_feats)
            out_bev_coors_lst.append(bev_coors)
            
        # Concatenate the pseudoimages along the batch dimension
        return torch.cat(pseudoimage_lst, dim=0), pt_fea_lst, out_bev_feats_lst, out_bev_coors_lst


class try_radar_DynamicEmbedder(nn.Module):
    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 in_channel: int, feat_channel: int) -> None:
        super().__init__()
        self.pseudo_image_dims = pseudo_image_dims
        
        # self.voxelizer = DynamicVoxelizer(voxel_size=voxel_size,
        #                                   point_cloud_range=point_cloud_range)
        
        self.feature_net = my_PillarFeatureNet(
            in_channels = in_channel,
            feat_channels = feat_channel, 
            point_cloud_range=point_cloud_range,
            voxel_size = voxel_size,
            mode='avg') # 'avg'
        
        self.scatter = my_PointPillarsScatter(in_channels=feat_channel, output_shape=pseudo_image_dims)
        
        self.dy_scatter1 = DynamicScatter(voxel_size,
                                            point_cloud_range,
                                            average_points=False)
        # self.dy_scatter2 = my_PointPillarsScatter(in_channels=1, output_shape=pseudo_image_dims)

    def forward(self, voxel_info_list) -> torch.Tensor:
    # def forward(self, points_in_fea, voxel_info_list) -> torch.Tensor:
        
        # voxel_info_list = self.voxelizer(points)

        pseudoimage_lst = []
        pt_fea_lst = []
        out_bev_feats_lst = []
        out_bev_coors_lst = []

        # dy_map_out = []
        dy_mask_out = []

        for batch_id, voxel_info_dict in enumerate(voxel_info_list):
            points = voxel_info_dict['points']
            points_dy = torch.abs(points[:,4]).reshape(points.shape[0],1) # N,1
            # voxel_size: [0.1, 0.4, 6]
            # coord: [ 0, 113, 322]
            coordinates = voxel_info_dict['voxel_coords']
            # my EDIT
            # valid_point_id = voxel_info_dict['point_idxes']
            # temp_pts_fea_batch = points_in_fea[batch_id]
            # temp_pts_fea = temp_pts_fea_batch[valid_point_id,:]

            # temp_pts_fea = points_in_fea[batch_id]
            temp_pts_fea = points

            dy_voxel_dys, dy_voxel_coors =  self.dy_scatter1(points_dy, coordinates)
            dy_voxel_dys = dy_voxel_dys > 0.1
            dy_voxel_dys = dy_voxel_dys.bool()
            # dy_map = self.dy_scatter2(dy_voxel_dys, dy_voxel_coors)
            # dy_map_out.append(dy_map)
            dy_mask_out.append(dy_voxel_dys.squeeze(1))

          
            # [ 0, 113, 325],
            voxel_feats, voxel_coors, point_feats = self.feature_net(points, temp_pts_fea, coordinates)
            # voxel_feats : torch.Size([NUM_non_empty_voxel, 32])
            # voxel_coors : torch.Size([NUM_non_empty_voxel, 3])   Example [ 0,  48, 331]
            # point_feats : torch.Size([ N_points , 32])

            pt_fea_lst.append(point_feats)
            
            bev_coors = voxel_coors[:,1:]
            pseudoimage = self.scatter(voxel_feats, voxel_coors)
            pseudoimage_lst.append(pseudoimage)

            out_bev_feats_lst.append(voxel_feats)
            out_bev_coors_lst.append(bev_coors)
            
        # Concatenate the pseudoimages along the batch dimension
        # return torch.cat(pseudoimage_lst, dim=0), pt_fea_lst, out_bev_feats_lst, out_bev_coors_lst, torch.cat(dy_map_out, dim=0), dy_mask_out
        return torch.cat(pseudoimage_lst, dim=0), pt_fea_lst, out_bev_feats_lst, out_bev_coors_lst, dy_mask_out


class my_Encoder(nn.Module):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__()

        assert out_channel>=in_channel
        self.en = False
        if out_channel>in_channel:
            self.mid = out_channel-in_channel
            self.encoder = nn.Linear(in_channel, self.mid, bias=False)
            self.en = True

        # self.encoder = nn.Sequential(
        #         nn.Linear(in_channel, out_channel, bias=False),
        #         nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01),
        #         nn.ReLU(inplace=True))
        

    def forward(self, voxel_info) -> torch.Tensor:
        pt_fea_lst = []
        for voxel_info_dict in voxel_info:
            points = voxel_info_dict['points']
            if self.en:           
                point_feats = self.encoder(points)
                point_feats = torch.cat((points, point_feats), dim=-1)
                pt_fea_lst.append(point_feats)
            else:
                pt_fea_lst.append(points)
            
        return pt_fea_lst
    




class conv_Encoder(nn.Module):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__()

        self.encoder = torch.nn.Conv1d(in_channel,out_channel,1)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        out = self.encoder(points.permute(0,2,1).contiguous())
            
        # Concatenate the pseudoimages along the batch dimension
        return out.permute(0,2,1)
    


class radar_Dymap(nn.Module):
    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range) -> None:
        super().__init__()
        # scatter 方法取max, 一个网格中只要有points_dy>1, 网格就会被标记为1
        self.scatter = DynamicScatter(voxel_size,
                                            point_cloud_range,
                                            average_points=False)
        self.out_scatter = my_PointPillarsScatter(in_channels=1, output_shape=pseudo_image_dims)

    def forward(self, voxel_info_list) -> torch.Tensor:
    # def forward(self, points_in_fea, voxel_info_list) -> torch.Tensor:
        
        # voxel_info_list = self.voxelizer(points)


        bev_dy_lst = []
        # bev_dy_coor = []

        for batch_id, voxel_info_dict in enumerate(voxel_info_list):
            points = voxel_info_dict['points']
            coordinates = voxel_info_dict['voxel_coords']

            # points_dy = torch.abs(points[:,4])> 0.1
            # # points_dy = torch.tensor(points_dy, dtype=torch.bool, device=points.device) # N,1
            # points_dy = points_dy.bool().reshape(points.shape[0],1)

            points_dy = torch.abs(points[:,4]).reshape(points.shape[0],1) # N,1
            
            voxel_dys, voxel_coors = self.scatter(points_dy, coordinates)

            voxel_dys = voxel_dys > 0.1
            # voxel_dys = voxel_dys.bool().reshape(voxel_dys.shape[0],1)
            voxel_dys = voxel_dys.bool()

            pseudoimage = self.out_scatter(voxel_dys, voxel_coors)

            # bev_dy_lst.append(voxel_dys)
            # bev_dy_coor.append(voxel_coors)
            bev_dy_lst.append(pseudoimage)

 
        # Concatenate the pseudoimages along the batch dimension
        # return bev_dy_lst, bev_dy_coor
        # return bev_dy_lst
        return torch.cat(bev_dy_lst, dim=0)




class cal_dy(nn.Module):
    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range) -> None:
        super().__init__()
        # self.vx = voxel_size[0]
        # self.vy = voxel_size[1]
        # self.vz = voxel_size[2]
        # self.point_cloud_range = point_cloud_range

        self.pseudo_image_dims = pseudo_image_dims
        
        self.scatter = DynamicScatter(voxel_size,
                                            point_cloud_range,
                                            average_points=False) 
        self.canvas_y = int(pseudo_image_dims[1])
        self.canvas_x = int(pseudo_image_dims[0])     
    
    def forward(self, li_input, li_voxel_info_list, ra_input, ra_voxel_info_list):
        li_dy = []
        li_ind = []
        ra_dy = []
        ra_ind = []

        for batch_id in range(len(li_voxel_info_list)):
            li_points = li_input[batch_id]  # (N_li_input, 3)
            cur_device = li_points.get_device()
            dy_li_mask = torch.zeros(li_points.shape[0], dtype=torch.bool, device=cur_device)
            li_ind_out = torch.zeros(li_points.shape[0], dtype=torch.int32, device=cur_device)
            
            li_coor = li_voxel_info_list[batch_id]['voxel_coords']
            li_valid = li_voxel_info_list[batch_id]['point_idxes']
            li_pts = li_points[li_valid] # 过滤不在voxelize区域内的输入点
            li_pts_ind = torch.zeros(li_pts.shape[0], dtype=torch.int32, device=cur_device)


            for pt_id in range(li_pts.shape[0]):
                temp_li_coor = li_coor[pt_id]
                assert temp_li_coor[0]==0
                indices = temp_li_coor[1] * self.canvas_x + temp_li_coor[2]
                li_pts_ind[pt_id] = indices.long()
            li_ind_out[li_valid] = li_pts_ind
            li_ind.append(li_ind_out)


            
            ra_points = ra_input[batch_id]  # (N_ra_input, 6)
            points_vr = torch.abs(ra_points[:,4])
            points_dy = points_vr > 0.1
            ra_dy.append(points_dy) # (N_ra_input, )
            ra_ind_out = torch.zeros(ra_points.shape[0], dtype=torch.int32, device=cur_device)

            ra_coor = ra_voxel_info_list[batch_id]['voxel_coords']
            ra_valid = ra_voxel_info_list[batch_id]['point_idxes']
            ra_pts = ra_points[ra_valid]
            ra_pts_ind = torch.zeros(ra_pts.shape[0], dtype=torch.int32, device=cur_device)

            for pt_id in range(ra_pts.shape[0]):
                temp_ra_coor = ra_coor[pt_id]
                assert temp_ra_coor[0]==0
                indices = temp_ra_coor[1] * self.canvas_x + temp_ra_coor[2]
                ra_pts_ind[pt_id] = indices.long()
            ra_ind_out[ra_valid] = ra_pts_ind
            ra_ind.append(ra_ind_out)

            points_vr = points_vr.reshape(ra_points.shape[0],1)
            voxel_dys, dy_coor = self.scatter(points_vr[ra_valid], ra_coor) 

            voxel_dys = voxel_dys > 0.1
            voxel_dys = voxel_dys.bool().squeeze(1)  # (R,)
            dy_ra_coor = dy_coor[voxel_dys] # 找到动态格子

            shift_tensor = gen_index_shift(2, cur_device) # 5*5
            grid_size = torch.tensor([self.pseudo_image_dims[0]+1, self.pseudo_image_dims[1]+1], device=cur_device)

            for id in range(dy_ra_coor.shape[0]):
                temp_dy_ra_coor = dy_ra_coor[id]
                for shift in shift_tensor:
                    shifted_coor = (temp_dy_ra_coor[1:] + shift) % grid_size
                    if (shifted_coor[0]<self.pseudo_image_dims[1]) and (shifted_coor[1]<self.pseudo_image_dims[0]):
                        cur_ind = shifted_coor[0] * self.canvas_x + shifted_coor[1]
                        dy_li_ind_mask = (li_ind_out==cur_ind)
                        dy_li_mask[dy_li_ind_mask] = True

            li_dy.append(dy_li_mask)
            

        return li_dy, li_ind, ra_dy, ra_ind
    



class ra_cluster(nn.Module):
    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range) -> None:
        super().__init__()
        # self.vx = voxel_size[0]
        # self.vy = voxel_size[1]
        # self.vz = voxel_size[2]
        # self.point_cloud_range = point_cloud_range

        self.pseudo_image_dims = pseudo_image_dims
        
        self.scatter = DynamicScatter(voxel_size,
                                            point_cloud_range,
                                            average_points=False) 
        self.canvas_y = int(pseudo_image_dims[1])
        self.canvas_x = int(pseudo_image_dims[0])     
    # pc0_voxel_infos_lst, radar_pc0_voxel_infos_lst, lidar_cluster
    def forward(self, li_voxel_info_list, ra_voxel_info_list, lidar_cluster, radar_ind):
        all_ra_cluster = []

        for batch_id in range(len(li_voxel_info_list)):
            
            cur_li_voxel_info_list = li_voxel_info_list[batch_id]
            cur_ra_voxel_info_list = ra_voxel_info_list[batch_id]
            
            cur_li_cluster = lidar_cluster[batch_id]
            cur_radar_ind = radar_ind[batch_id]
            cur_device = cur_radar_ind.get_device()
            ra_cluster = torch.zeros(cur_radar_ind.shape[0], dtype=torch.int16, device=cur_device)
           

            li_coor = cur_li_voxel_info_list['voxel_coords']
            li_valid = cur_li_voxel_info_list['point_idxes']
            valid_li_cluster = cur_li_cluster[li_valid]
            valid_li_cluster = valid_li_cluster.float().reshape(-1,1)

            bev_cluster_class, bev_cluster_coor = self.scatter(valid_li_cluster.float(), li_coor) 

            for id in range(bev_cluster_coor.shape[0]):
                temp_coor = bev_cluster_coor[id]
                cur_ind = temp_coor[1] * self.canvas_x + temp_coor[2]
                ra_grid_mask = (cur_radar_ind == cur_ind)
                ra_cluster[ra_grid_mask] = bev_cluster_class[id].short()
            
            all_ra_cluster.append(ra_cluster)

        return all_ra_cluster