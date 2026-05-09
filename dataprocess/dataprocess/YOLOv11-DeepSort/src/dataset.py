"""
# Created: 2023-11-04 15:52
# Updated: 2024-07-12 23:16
# 
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/), Jaeyeul Kim (jykim94@dgist.ac.kr)
#
# Change Logs:
# 2024-11-06: Added Data Augmentation transform for RandomHeight, RandomFlip, RandomJitter from DeltaFlow project.
# 2024-07-12: Merged num_frame based on Flow4D model from Jaeyeul Kim.
# 
# Description: Torch dataloader for the dataset we preprocessed.
# 
# This file is part of 
# * OpenSceneFlow (https://github.com/KTH-RPL/OpenSceneFlow)
# 
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
"""

import torch, re
from torch.utils.data import Dataset, DataLoader
import h5py, pickle, argparse
from tqdm import tqdm
import numpy as np
from torchvision import transforms
import random

import os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from src.utils import import_func
from src.augmentations import RandomHeight, RandomFlip, RandomJitter, ToTensor

def extract_flow_number(key):
    digits = re.findall(r'\d+$', key)
    if digits:
        return digits[0]
    return '0'

# TODO: clean up
def collate_fn_pad(batch):

    num_frames = 2
    while f'pch{num_frames - 1}' in batch[0]:
        num_frames += 1

    pc0_after_mask_ground, pc1_after_mask_ground = [], []
    beam0_after_mask_ground, beam1_after_mask_ground = [], []
    pch_after_mask_ground = [[] for _ in range(num_frames - 2)]
    beamh_after_mask_ground = [[] for _ in range(num_frames - 2)]

    for i in range(len(batch)):
        gm0 = batch[i]['gm0']
        gm1 = batch[i]['gm1']

        pc0_after_mask_ground.append(batch[i]['pc0'][~gm0])
        pc1_after_mask_ground.append(batch[i]['pc1'][~gm1])

        if 'beam0' in batch[i]:
            beam0_after_mask_ground.append(batch[i]['beam0'][~gm0])
        if 'beam1' in batch[i]:
            beam1_after_mask_ground.append(batch[i]['beam1'][~gm1])

        for j in range(1, num_frames - 1):
            p_gm = batch[i][f'gmh{j}']
            p_pc = batch[i][f'pch{j}'][~p_gm]
            pch_after_mask_ground[j-1].append(p_pc)
            beamh_key = f'beamh{j}'
            if beamh_key in batch[i]:
                beamh_after_mask_ground[j-1].append(batch[i][beamh_key][~p_gm])

    pc0_after_mask_ground = torch.nn.utils.rnn.pad_sequence(
        [torch.as_tensor(x) for x in pc0_after_mask_ground],
        batch_first=True, padding_value=torch.nan
    )
    pc1_after_mask_ground = torch.nn.utils.rnn.pad_sequence(
        [torch.as_tensor(x) for x in pc1_after_mask_ground],
        batch_first=True, padding_value=torch.nan
    )
    pch_after_mask_ground = [
        torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x) for x in pch_],
            batch_first=True, padding_value=torch.nan
        )
        for pch_ in pch_after_mask_ground
    ]

    def pad_beam_list(lst):
        if len(lst) == 0:
            return None
        lst_t = [torch.as_tensor(x, dtype=torch.uint8) for x in lst]
        return torch.nn.utils.rnn.pad_sequence(
            lst_t, batch_first=True, padding_value=255
        )
    beam0_padded = pad_beam_list(beam0_after_mask_ground)
    beam1_padded = pad_beam_list(beam1_after_mask_ground)
    beamh_padded = [pad_beam_list(bh) for bh in beamh_after_mask_ground]

    res_dict =  {
        'pc0': pc0_after_mask_ground,
        'pc1': pc1_after_mask_ground,
        'pose0': [batch[i]['pose0'] for i in range(len(batch))],
        'pose1': [batch[i]['pose1'] for i in range(len(batch))],
    }

    if beam0_padded is not None: res_dict['beam0'] = beam0_padded
    if beam1_padded is not None: res_dict['beam1'] = beam1_padded

    for j in range(1, num_frames - 1):
        res_dict[f'pch{j}'] = pch_after_mask_ground[j-1]
        res_dict[f'poseh{j}'] = [batch[i][f'poseh{j}'] for i in range(len(batch))]
        if beamh_padded[j-1] is not None:
            res_dict[f'beamh{j}'] = beamh_padded[j-1]

    if 'flow' in batch[0]:
        flow = torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(batch[i]['flow'][~batch[i]['gm0']]) for i in range(len(batch))],
            batch_first=True
        )
        flow_is_valid = torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(batch[i]['flow_is_valid'][~batch[i]['gm0']]) for i in range(len(batch))],
            batch_first=True
        )
        flow_category_indices = torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(batch[i]['flow_category_indices'][~batch[i]['gm0']]) for i in range(len(batch))],
            batch_first=True
        )
        res_dict['flow'] = flow
        res_dict['flow_is_valid'] = flow_is_valid
        res_dict['flow_category_indices'] = flow_category_indices

    if 'ego_motion' in batch[0]:
        res_dict['ego_motion'] = [batch[i]['ego_motion'] for i in range(len(batch))]

    if 'pc0_dynamic' in batch[0]:
        pc0_dyn = torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(batch[i]['pc0_dynamic'][~batch[i]['gm0']]) for i in range(len(batch))],
            batch_first=True, padding_value=0
        )
        pc1_dyn = torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(batch[i]['pc1_dynamic'][~batch[i]['gm1']]) for i in range(len(batch))],
            batch_first=True, padding_value=0
        )
        res_dict['pc0_dynamic'] = pc0_dyn
        res_dict['pc1_dynamic'] = pc1_dyn

    return res_dict

class HDF5Dataset(Dataset):
    def __init__(self, directory, \
                transform=None, n_frames=2, ssl_label=None, \
                eval = False, leaderboard_version=1, \
                vis_name='', scene_fraction: float = None):
        '''
        Args:
            directory: the directory of the dataset, the folder should contain some .h5 file and index_total.pkl.

            Following are optional:
            * transform: for data augmentation, default is None.
            * n_frames: the number of frames we use, default is 2: current (pc0), next (pc1); if it's more than 2, then it read the history from current.
            * ssl_label: if attr, it will read the dynamic cluster label. Otherwise, no dynamic cluster label in data dict.
            * eval: if True, use the eval index (only used it for leaderboard evaluation)
            * leaderboard_version: 1st or 2nd, default is 1. If '2', we will use the index_eval_v2.pkl from assets/docs.
            * vis_name: the data of the visualization, default is ''.
            * scene_fraction: the fraction of total scenes it will use
        '''
        super(HDF5Dataset, self).__init__()
        self.directory = directory
        if (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or not torch.distributed.is_initialized():
            print(f"----[Debug] Loading data with num_frames={n_frames}, ssl_label={ssl_label}, eval={eval}, leaderboard_version={leaderboard_version}")
        with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f:
            self.data_index = pickle.load(f)

        self.eval_index = False
        self.ssl_label = import_func(f"src.autolabel.{ssl_label}") if ssl_label is not None else None
        self.history_frames = n_frames - 2
        self.vis_name = vis_name if isinstance(vis_name, list) else [vis_name]
        self.transform = transform

        if eval:
            eval_index_file = os.path.join(self.directory, 'index_eval.pkl')
            if leaderboard_version == 2:
                print("Using index to leaderboard version 2!!")
                eval_index_file = os.path.join(BASE_DIR, 'assets/docs/index_eval_v2.pkl')

            if not os.path.exists(eval_index_file):
                print(f"Warning: No {eval_index_file} file found! We will try {'index_flow.pkl'}")
                eval_index_file = os.path.join(self.directory, 'index_flow.pkl')
                if not os.path.exists(eval_index_file):
                    raise Exception(f"No any eval index file found! Please check {self.directory}")
            
            self.eval_index = eval
            with open(eval_index_file, 'rb') as f:
                self.eval_data_index = pickle.load(f)

        self.scene_id_bounds = {}  # 存储每个scene_id的最大最小timestamp和位置
        for idx, (scene_id, timestamp) in enumerate(self.data_index):
            if scene_id not in self.scene_id_bounds:
                self.scene_id_bounds[scene_id] = {
                    "min_timestamp": timestamp, "max_timestamp": timestamp,
                    "min_index": idx, "max_index": idx
                }
            else:
                bounds = self.scene_id_bounds[scene_id]
                if timestamp < bounds["min_timestamp"]:
                    bounds["min_timestamp"] = timestamp
                    bounds["min_index"] = idx
                if timestamp > bounds["max_timestamp"]:
                    bounds["max_timestamp"] = timestamp
                    bounds["max_index"] = idx
        
        # for some dataset that annotated HZ is different.... like truckscene and nuscene etc.
        self.train_index = None
        if not eval and ssl_label is None and transform is not None: # transform indicates whether we are in training mode.
            # check if train seq all have gt.
            one_scene_id = list(self.scene_id_bounds.keys())[0]
            check_flow_exist = True
            with h5py.File(os.path.join(self.directory, f'{one_scene_id}.h5'), 'r') as f:
                for i in range(self.scene_id_bounds[one_scene_id]["min_index"], self.scene_id_bounds[one_scene_id]["max_index"]):
                        scene_id, timestamp = self.data_index[i]
                        key = str(timestamp)
                        if 'flow' not in f[key]:
                            check_flow_exist = False
                            break
            if not check_flow_exist:
                print(f"----- [Warning]: Not all frames have flow data, we will instead use the index_flow.pkl to train.")
                self.train_index = pickle.load(open(os.path.join(self.directory, 'index_flow.pkl'), 'rb'))
                
        if (scene_fraction is not None) and (not eval):
            if self.train_index is None:
                self.train_index = list(self.data_index)

            all_scenes = sorted({sid for (sid, _) in self.train_index})
            target_n = max(1, int(len(all_scenes) * float(scene_fraction)))
            keep = set(random.sample(all_scenes, target_n))

            before = len(self.train_index)
            self.train_index = [(sid, ts) for (sid, ts) in self.train_index if sid in keep]
            after = len(self.train_index)

            print(f"[scene_fraction] {scene_fraction:.3f}: {before} -> {after} frames across {len(keep)} scenes")

    def __len__(self):
        # return 100 # for testing
        if self.eval_index:
            return len(self.eval_data_index)
        elif not self.eval_index and self.train_index is not None:
            return len(self.train_index)
        return len(self.data_index)
    
    def valid_index(self, index_):
        """
        Check if the index is valid for the current mode and satisfy the constraints.
        """
        eval_flag = False
        if self.eval_index:
            eval_index_ = index_
            scene_id, timestamp = self.eval_data_index[eval_index_]
            index_ = self.data_index.index([scene_id, timestamp])
            max_idx = self.scene_id_bounds[scene_id]["max_index"]
            if index_ >= max_idx:
                _, index_ = self.valid_index(eval_index_ - 1)
            eval_flag = True
        elif self.train_index is not None:
            train_index_ = index_
            scene_id, timestamp = self.train_index[train_index_]
            max_idx = self.scene_id_bounds[scene_id]["max_index"]
            index_ = self.data_index.index([scene_id, timestamp])
            if index_ >= max_idx:
                _, index_ = self.valid_index(train_index_ - 1)
        else:
            scene_id, timestamp = self.data_index[index_]
            max_idx = self.scene_id_bounds[scene_id]["max_index"]
            min_idx = self.scene_id_bounds[scene_id]["min_index"]

            max_valid_index_for_flow = max_idx - 1
            min_valid_index_for_flow = min_idx + self.history_frames
            index_ = max(min_valid_index_for_flow, min(max_valid_index_for_flow, index_))
        return eval_flag, index_
    
    def __getitem__(self, index_):
        eval_flag, index_ = self.valid_index(index_)
        scene_id, timestamp = self.data_index[index_]

        key = str(timestamp)
        data_dict = {
            'scene_id': scene_id,
            'timestamp': timestamp,
            'eval_flag': eval_flag
        }
        with h5py.File(os.path.join(self.directory, f'{scene_id}.h5'), 'r') as f:
            # original data
            data_dict['pc0'] = f[key]['lidar'][:][:,:3]
            data_dict['gm0'] = f[key]['ground_mask'][:]
            data_dict['pose0'] = f[key]['pose'][:]
            if self.ssl_label is not None:
                data_dict['pc0_dynamic'] = self.ssl_label(f[key])
            if 'beam_id' in f[key]:
                data_dict['beam0'] = f[key]['beam_id'][:].astype(np.uint8)

            if self.history_frames >= 0: 
                next_timestamp = str(self.data_index[index_ + 1][1])
                data_dict['pose1'] = f[next_timestamp]['pose'][:]
                data_dict['pc1'] = f[next_timestamp]['lidar'][:][:,:3]
                data_dict['gm1'] = f[next_timestamp]['ground_mask'][:]
                if self.ssl_label is not None:
                    data_dict['pc1_dynamic'] = self.ssl_label(f[next_timestamp])
                if 'beam_id' in f[next_timestamp]:
                    data_dict['beam1'] = f[next_timestamp]['beam_id'][:].astype(np.uint8)
                past_frames = []
                for i in range(1, self.history_frames + 1):
                    frame_index = index_ - i
                    if frame_index < self.scene_id_bounds[scene_id]["min_index"]: 
                        frame_index = self.scene_id_bounds[scene_id]["min_index"] 

                    past_timestamp = str(self.data_index[frame_index][1])
                    past_pc = f[past_timestamp]['lidar'][:][:,:3]
                    past_gm = f[past_timestamp]['ground_mask'][:]
                    past_pose = f[past_timestamp]['pose'][:]

                    past_frames.append((past_pc, past_gm, past_pose))
                    if i == 1 and self.ssl_label is not None: # only for history 1: t-1
                        # data_dict['pch1_dynamic'] = f[past_timestamp]['label'][:].astype('int16')
                        data_dict['pch1_dynamic'] = self.ssl_label(f[past_timestamp])
                    if 'beam_id' in f[past_timestamp]:
                        data_dict[f'beamh{i}'] = f[past_timestamp]['beam_id'][:].astype(np.uint8)
                for i, (past_pc, past_gm, past_pose) in enumerate(past_frames):
                    data_dict[f'pch{i+1}'] = past_pc
                    data_dict[f'gmh{i+1}'] = past_gm
                    data_dict[f'poseh{i+1}'] = past_pose

            for data_key in self.vis_name + ['ego_motion', 'lidar_dt', 
                             # ground truth information:
                             'flow', 'flow_is_valid', 'flow_category_indices', 'flow_instance_id', 'dufo']:
                if data_key in f[key]:
                    data_dict[data_key] = f[key][data_key][:]

            if self.eval_index:
                # looks like v2 not follow the same rule as v1 with eval_mask provided
                # data_dict['eval_mask'] = np.ones_like(data_dict['pc0'][:, 0], dtype=np.bool_) if 'eval_mask' not in f[key] else f[key]['eval_mask'][:]
                if 'eval_mask' in f[key]:
                    data_dict['eval_mask'] = f[key]['eval_mask'][:]
                elif 'ground_mask' in f[key]:
                    data_dict['eval_mask'] = ~f[key]['ground_mask'][:]
                else:
                    data_dict['eval_mask'] = np.ones_like(data_dict['pc0'][:, 0], dtype=np.bool_)
                    
        if self.transform:
            data_dict = self.transform(data_dict)
        return data_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DataLoader test")
    parser.add_argument('--data_mode', '-m', type=str, default='train', metavar='N', help='Dataset mode.')
    parser.add_argument('--data_dir', '-d', type=str, default='/home/kin/data/av2/h5py_v2/sensor', metavar='N', help='preprocess data path.')
    options = parser.parse_args()

    # testing eval mode
    dataset = HDF5Dataset(directory = options.data_dir+"/"+options.data_mode, eval = False,
                          transform = transforms.Compose([RandomHeight(), RandomFlip(), RandomJitter(), ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=16, collate_fn=collate_fn_pad)
    for data in tqdm(dataloader, ncols=80, desc="read data mode"):
        res_dict = data
        # print(res_dict['pc0'].shape)
        # break