"""
# Created: 2023-11-04 15:52
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Description: Torch dataloader for the dataset we preprocessed.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py, os, pickle, argparse
from tqdm import tqdm
import numpy as np



def radarflow_vod_collate_fn_pad(batch):
    npoints = 256

    pc0_after_mask_ground, pc1_after_mask_ground = [], []
    radar_pc0, radar_pc1, radar_ft0, radar_ft1 = [], [], [], []
    radar_flow, radar_flow_is_valid, radar_flow_category_indices, radar_motion_mask, clean_radar_mask = [], [], [], [], []
    pc0_vr_c, pc1_vr_c =  [], []
    radar_id, pc1_radar_id = [], []

    ds_id_seg, pc1_ds_id_seg = [], []
    
    for i in range(len(batch)):
        pc0_after_mask_ground.append(batch[i]['lidar_pc0'])
        pc1_after_mask_ground.append(batch[i]['lidar_pc1'])

        temp_radar_ft0 = torch.cat([batch[i]['radar0_vr'], batch[i]['radar0_RCS']], dim=1)
        temp_radar_ft1 = torch.cat([batch[i]['radar1_vr'], batch[i]['radar1_RCS']], dim=1)

        npts_1 = temp_radar_ft0.shape[0]
        npts_2 = temp_radar_ft1.shape[0]

        if npts_1<npoints:
            sample_idx1 = np.arange(0,npts_1)
            sample_idx1 = np.append(sample_idx1, np.random.choice(npts_1,npoints-npts_1,replace=True))
        else:
            sample_idx1 = np.random.choice(npts_1, npoints, replace=False)

        if npts_2<npoints:
            sample_idx2 = np.arange(0,npts_2)
            sample_idx2 = np.append(sample_idx2, np.random.choice(npts_2,npoints-npts_2,replace=True))
        else:
            sample_idx2 = np.random.choice(npts_2, npoints, replace=False)

        sample_idx1 = torch.tensor(sample_idx1)
        sample_idx2 = torch.tensor(sample_idx2)

        radar_ft0.append(temp_radar_ft0[sample_idx1,:].unsqueeze(0))
        radar_ft1.append(temp_radar_ft1[sample_idx2,:].unsqueeze(0))

        radar_pc0.append(batch[i]['radar_pc0'][sample_idx1,:].unsqueeze(0)) 
        radar_pc1.append(batch[i]['radar_pc1'][sample_idx2,:].unsqueeze(0))

        radar_flow.append(batch[i]['radar_flow'][sample_idx1,:].unsqueeze(0)) 
        radar_flow_is_valid.append(batch[i]['radar_flow_valid'][sample_idx1].unsqueeze(0))
        radar_flow_category_indices.append(batch[i]['radar_flow_category'][sample_idx1].unsqueeze(0))
        radar_motion_mask.append(batch[i]['radar_motion_mask'][sample_idx1].unsqueeze(0))
        clean_radar_mask.append(batch[i]['clean_radar_mask'][sample_idx1].unsqueeze(0))

        radar_id.append(batch[i]['radar_id'][sample_idx1].unsqueeze(0))
        pc1_radar_id.append(batch[i]['pc1_radar_id'][sample_idx2].unsqueeze(0))
        pc0_vr_c.append(batch[i]['radar0_vr_compensated'][sample_idx1].unsqueeze(0))
        pc1_vr_c.append(batch[i]['radar1_vr_compensated'][sample_idx2].unsqueeze(0))

        ds_id_seg.append(batch[i]['ds_id_seg'][sample_idx1].unsqueeze(0))
        pc1_ds_id_seg.append(batch[i]['pc1_ds_id_seg'][sample_idx2].unsqueeze(0))
        
    pc0_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pc0_after_mask_ground, batch_first=True, padding_value=torch.nan)
    pc1_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pc1_after_mask_ground, batch_first=True, padding_value=torch.nan)

    radar_ft0 = torch.cat(radar_ft0, dim = 0)
    radar_ft1 = torch.cat(radar_ft1, dim = 0)
    radar_pc0 = torch.cat(radar_pc0, dim = 0)
    radar_pc1 = torch.cat(radar_pc1, dim = 0)

    radar_id = torch.cat(radar_id, dim = 0)
    pc1_radar_id = torch.cat(pc1_radar_id, dim = 0)
    pc0_vr_c = torch.cat(pc0_vr_c, dim = 0)
    pc1_vr_c = torch.cat(pc1_vr_c, dim = 0)

    ds_id_seg = torch.cat(ds_id_seg, dim = 0)
    pc1_ds_id_seg = torch.cat(pc1_ds_id_seg, dim = 0)



    if 'radar_flow' in batch[0]:
        radar_flow = torch.cat(radar_flow, dim = 0)
        radar_flow_is_valid = torch.cat(radar_flow_is_valid, dim = 0)
        radar_flow_category_indices = torch.cat(radar_flow_category_indices, dim = 0)
        radar_motion_mask = torch.cat(radar_motion_mask, dim = 0)
        clean_radar_mask = torch.cat(clean_radar_mask, dim = 0)
    

    
    res_dict =  {
        'pc0': pc0_after_mask_ground,
        'pc1': pc1_after_mask_ground,
        'radar_pc0' : radar_pc0,
        'radar_pc1' : radar_pc1,
        'radar_ft0' : radar_ft0,
        'radar_ft1' : radar_ft1,

        'pose0': [batch[i]['pose0'] for i in range(len(batch))],
        'pose1': [batch[i]['pose1'] for i in range(len(batch))],

        'radar_flow' : radar_flow,
        'radar_flow_is_valid' : radar_flow_is_valid,
        'radar_flow_category_indices' : radar_flow_category_indices,
        'radar_motion_mask' : radar_motion_mask,
        'clean_radar_mask': clean_radar_mask,


        "radar_id" : radar_id,
        "pc1_radar_id" : pc1_radar_id,
        "pc0_vr_c": pc0_vr_c,
        "pc1_vr_c": pc1_vr_c,

        "ds_id_seg" : ds_id_seg,
        "pc1_ds_id_seg" : pc1_ds_id_seg
    }


    if 'ego_motion' in batch[0]:
        res_dict['ego_motion'] = [batch[i]['ego_motion'] for i in range(len(batch))]




    return res_dict






class radarflow_VODH5Dataset(Dataset):
    def __init__(self, directory, mode):
        '''
        directory: the directory of the dataset
        eval: if True, use the eval index
        '''
        super(radarflow_VODH5Dataset, self).__init__()
        self.directory = directory + mode
     
        with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f:
            self.data_index = pickle.load(f)
               
        self.scene_id_bounds = {} 
        for idx, (scene_id, timestamp) in enumerate(self.data_index):
            if scene_id not in self.scene_id_bounds:
                self.scene_id_bounds[scene_id] = {
                    "min_timestamp": int(timestamp),
                    "max_timestamp": int(timestamp),
                    "min_index": idx,
                    "max_index": idx
                }
            else:
                bounds = self.scene_id_bounds[scene_id]
                if int(timestamp) < bounds["min_timestamp"]:
                    bounds["min_timestamp"] = int(timestamp)
                    bounds["min_index"] = idx
                if int(timestamp) > bounds["max_timestamp"]:
                    bounds["max_timestamp"] = int(timestamp)
                    bounds["max_index"] = idx

    def __len__(self):
        return len(self.data_index)
           
    def __getitem__(self, index_):
        scene_id, timestamp = self.data_index[index_]
        # to make sure we have continuous frames
        if self.scene_id_bounds[scene_id]["max_index"] == index_:
            index_ = index_ - 1
        scene_id, timestamp = self.data_index[index_]


        key = str(timestamp)

        with h5py.File(os.path.join(self.directory, f'{scene_id}.h5'), 'r') as f:
            lidar_pc0 = torch.tensor(f[key]['lidar_pc'][:,:3])
            radar_pc0 = torch.tensor(f[key]['radar_pc'][:,:3])
            radar0_RCS = torch.tensor(f[key]['radar_pc'][:,3].reshape(-1,1)) # Radar Cross Section
            radar0_vr = torch.tensor(f[key]['radar_pc'][:,4].reshape(-1,1))
            radar0_vr_compensated = torch.tensor(f[key]['radar_pc'][:,5].reshape(-1,1))
            radar0_time = torch.tensor(f[key]['radar_pc'][:,6].reshape(-1,1))
            pose0 = torch.tensor(f[key]['pose'][:])

            next_timestamp = str(self.data_index[index_+1][1])   
            lidar_pc1 = torch.tensor(f[next_timestamp]['lidar_pc'][:,:3])
            radar_pc1 = torch.tensor(f[next_timestamp]['radar_pc'][:,:3])
            radar1_RCS = torch.tensor(f[next_timestamp]['radar_pc'][:,3].reshape(-1,1)) # Radar Cross Section
            radar1_vr = torch.tensor(f[next_timestamp]['radar_pc'][:,4].reshape(-1,1))
            radar1_vr_compensated = torch.tensor(f[next_timestamp]['radar_pc'][:,5].reshape(-1,1))
            radar1_time = torch.tensor(f[next_timestamp]['radar_pc'][:,6].reshape(-1,1))
            pose1 = torch.tensor(f[next_timestamp]['pose'][:])

            res_dict = {
                'scene_id': scene_id,
                'timestamp': key,

                'lidar_pc0': lidar_pc0,
                'radar_pc0': radar_pc0,
                'radar0_RCS': radar0_RCS,
                'radar0_vr_compensated': radar0_vr_compensated,
                'radar0_vr': radar0_vr,
                'radar0_time' : radar0_time,
                'pose0': pose0,

                'lidar_pc1': lidar_pc1,
                'radar_pc1': radar_pc1,
                'radar1_RCS': radar1_RCS,
                'radar1_vr_compensated': radar1_vr_compensated,
                'radar1_vr': radar1_vr,
                'radar1_time' : radar1_time,
                'pose1': pose1,
            }

                
            if 'radar_flow' in f[key]:
                radar_flow = torch.tensor(f[key]['radar_flow'][:])
                radar_flow_valid = torch.tensor(f[key]['radar_flow_valid'][:])
                radar_flow_category = torch.tensor(f[key]['radar_flow_category'][:])
                radar_motion_mask = torch.tensor(f[key]['radar_motion_mask'][:])
                clean_radar_mask = torch.tensor(f[key]['clean_radar_mask'][:])

                res_dict['radar_flow'] = radar_flow
                res_dict['radar_flow_valid'] = radar_flow_valid
                res_dict['radar_flow_category'] = radar_flow_category
                res_dict['radar_motion_mask'] = radar_motion_mask
                res_dict['clean_radar_mask'] = clean_radar_mask

            if 'pc1_radar_motion_mask' in f[key]:
                pc1_radar_motion_mask = torch.tensor(f[key]['pc1_radar_motion_mask'][:])
                res_dict['pc1_radar_motion_mask'] = pc1_radar_motion_mask


            if 'ego_motion' in f[key]:
                ego_motion = torch.tensor(f[key]['ego_motion'][:])
                res_dict['ego_motion'] = ego_motion

          
            if "radar_id" in f[key]:
                res_dict['radar_id'] = torch.tensor(f[key]["radar_id"][:].astype(int))
                res_dict['pc1_radar_id'] = torch.tensor(f[next_timestamp]["radar_id"][:].astype(int)) 

                    
            if 'yolo11_ds_id_seg' in f[key]:
                res_dict['ds_id_seg'] = torch.tensor(f[key]["yolo11_ds_id_seg"][:].astype(int))   # v2_ds_id_seg  ds_id_seg
                res_dict['pc1_ds_id_seg'] = torch.tensor(f[next_timestamp]["yolo11_ds_id_seg"][:].astype(int))  # v2_ds_id_seg  ds_id_seg
   
        return res_dict



class preprocess_VODH5Dataset(Dataset):
    def __init__(self, directory):
        '''
        directory: the directory of the dataset
        eval: if True, use the eval index
        '''
        super(preprocess_VODH5Dataset, self).__init__()
        self.directory = directory
        
        with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f:
            self.data_index = pickle.load(f)
               
        self.scene_id_bounds = {}  
        for idx, (scene_id, timestamp) in enumerate(self.data_index):
            if scene_id not in self.scene_id_bounds:
                self.scene_id_bounds[scene_id] = {
                    "min_timestamp": int(timestamp),
                    "max_timestamp": int(timestamp),
                    "min_index": idx,
                    "max_index": idx
                }
            else:
                bounds = self.scene_id_bounds[scene_id]
                if int(timestamp) < bounds["min_timestamp"]:
                    bounds["min_timestamp"] = int(timestamp)
                    bounds["min_index"] = idx
                if int(timestamp) > bounds["max_timestamp"]:
                    bounds["max_timestamp"] = int(timestamp)
                    bounds["max_index"] = idx

    def __len__(self):
        return len(self.data_index)
    
    def __getitem__(self, index_):
        scene_id, timestamp = self.data_index[index_]

        key = str(timestamp)
        cur_path = os.path.join(self.directory, f'{scene_id}.h5')
        with h5py.File(cur_path, 'r') as f:
            print(cur_path)
            lidar_pc0 = f[key]['lidar_pc'][:,:3]
            radar_pc0 = f[key]['radar_pc'][:,:3]
            pose0 = f[key]['pose'][:]

            res_dict = {
                'scene_id': scene_id,
                'timestamp': key,

                'lidar_pc0': lidar_pc0,
                'radar_pc0': radar_pc0,
                'pose0': pose0,
            }

            if "radar_motion_mask" in f[key]:
                res_dict['radar_motion_mask'] = f[key]["radar_motion_mask"][:]

            if "radar_flow_category" in f[key]:
                res_dict['radar_flow_category'] = f[key]["radar_flow_category"][:]

            if "radar_id" in f[key]:
                res_dict['radar_id'] = f[key]["radar_id"][:]

        return res_dict




