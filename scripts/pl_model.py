import torch.nn as nn
import torch
import numpy as np
import os
# import torch.nn.functional as F

from scripts.cmflow_utils.model_utils import *
from scripts.cmflow_utils import *

import torch.optim as optim
from pathlib import Path
from lightning import LightningModule
from hydra.utils import instantiate
from omegaconf import OmegaConf    # , open_dict
import sys, time, h5py
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from scripts.utils.mics import import_func, weights_init  #  zip_res
# from scripts.utils.av2_eval import write_output_file
from scripts.network.models.basic import cal_pose0to1
from scripts.network.official_metric import evaluate_leaderboard, evaluate_leaderboard_v2
from scripts.network.my_evaluation_metric import new_OfficialMetrics , evaluate_leaderboard_v3
from scripts.network.radar_loss import *


class ModelWrapper(LightningModule):
    def __init__(self,cfg): # 256 220
        super().__init__()

        OmegaConf.set_struct(cfg.model.target, True)
        self.zeta = 0.005
        
        self.output_dir = None
        if 'my_model_save_path' in cfg:
            self.output_dir = cfg.my_model_save_path + cfg.model_name + '/temp'

        if 'av2_mode' in cfg:
            self.av2_mode = cfg.av2_mode
        else:
            self.av2_mode = None
        self.npoints = 256
        self.interval = 0.10
        if 'val_every' in cfg:
            self.val_every = cfg.val_every

        self.model = instantiate(cfg.model.target)
        self.model.apply(weights_init)

        ############################################
        ############################################
        # Lstat
        self.rastat_Loss = True # True False
        if self.rastat_Loss:
            self.ra_loss_fn = import_func("scripts.network.loss_func.PT_raLoss")

        # Lic
        self.cham_Loss = True # True False

        # Lis
        self.box_loss = True # True False
        if self.box_loss:
            self.box_loss_fn = import_func("scripts.network.loss_func.dy_seg_Loss_mean")

        ############################################
        ############################################
        

        if 'pretrained_weights' in cfg:
            if cfg.pretrained_weights is not None:
                self.model.load_from_checkpoint(cfg.pretrained_weights)
                print("Successfully load model parameters!")

        self.batch_size = int(cfg.batch_size) if 'batch_size' in cfg else 1
        self.lr = cfg.lr if 'lr' in cfg else None
        self.epochs = cfg.epochs if 'epochs' in cfg else None
        self.decay_epochs = cfg.decay_epochs if 'decay_epochs' in cfg else None
        self.decay_rate = cfg.decay_rate if 'decay_rate' in cfg else None

        self.radar_metrics =  new_OfficialMetrics(class_name= "Radar") 

        if 'checkpoint' in cfg:
            self.load_checkpoint_path = cfg.checkpoint

        if "vod_mode" in cfg:
            self.vod_mode = cfg.vod_mode

        if 'dataset_path' in cfg:
            self.dataset_path = cfg.dataset_path

        self.save_hyperparameters()
        self.temp_epoch = 0

    def training_step(self, batch, batch_idx):
        total_loss = 0.0
        res_dict = self.model(batch)

        ###################################
        ##   Results from model output   ##
        ###################################
        batch_sizes = len(batch["pose0"])
        radar_pose_flows = res_dict['radar_pose_flow']
        radar_est_flow = res_dict['radar_flow']
        
        ######################
        ##   From Dataset   ##
        ######################

        # GT radar_id
        gt_pc0_track_id = batch['radar_id'] # B N
        gt_pc1_track_id = batch['pc1_radar_id'] # B N

        radar_pc0s = batch['radar_pc0']   # B N 3
        radar_pc1s = batch['radar_pc1']   # B N 3

        radar_motion_mask = batch['radar_motion_mask']

        pc0_ds_id_seg = batch['ds_id_seg']
        pc1_ds_id_seg = batch['pc1_ds_id_seg']

        radar_gt_flow = batch['radar_flow']
        ra_pc0_dys = torch.abs(batch['pc0_vr_c'].squeeze(2)) > 0.1 # B N 1  --> B N

        ############################################
        ############################################

        # First
        if self.rastat_Loss:
            for batch_id in range(batch_sizes):
                radar_est_flow_ = radar_est_flow[batch_id]
                radar_pose_flow_ = radar_pose_flows[batch_id]

                radar_dict2loss = {'est_flow': radar_est_flow_,
                                "pc0": radar_pc0s[batch_id],  
                                "pc1": radar_pc1s[batch_id],
                                'ra_pc0_dy': ra_pc0_dys[batch_id],
                                'pose_flow': radar_pose_flow_}   
                        
                loss_rastat =  self.ra_loss_fn(radar_dict2loss)
                total_loss = total_loss + loss_rastat


        # Second
        if self.cham_Loss:
            cham_loss_obj = ID_ChamferLoss() 
            for batch_id in range(batch_sizes):
                pc0_id_seg = pc0_ds_id_seg[batch_id]
                pc1_id_seg = pc1_ds_id_seg[batch_id]
             
                loss_cham  = cham_loss_obj(radar_pc0s[batch_id], radar_pc1s[batch_id], radar_est_flow[batch_id],
                                           pc0_id_seg, pc1_id_seg)
                total_loss = total_loss + loss_cham


        # Third
        if self.box_loss:
            gt_radar_ids = pc0_ds_id_seg   

            for batch_id in range(batch_sizes):
                radar_pose_flow_ = radar_pose_flows[batch_id]
                pc0_gt_radar_id = gt_radar_ids[batch_id]
                radar_est_flow_ = radar_est_flow[batch_id]
                dy_seg_res = {'est_flow': radar_est_flow_ - radar_pose_flow_,
                            'track_id': pc0_gt_radar_id,
                            'pose_flow': radar_pose_flow_,
                            "pc0": radar_pc0s[batch_id]
                            }
                loss2 = self.box_loss_fn(dy_seg_res)
                total_loss = total_loss + loss2
        
        self.log("trainer/loss", total_loss/batch_sizes, sync_dist=True, batch_size=self.batch_size)
        return total_loss/batch_sizes
        
    
    def train_validation_step_(self, batch, res_dict):
        # means there are ground truth flow so we can evaluate the EPE-3 Way metric
        if (batch['radar_flow'][0].shape[0] > 0):
            radar_pose_flows = res_dict['radar_pose_flow']

            for batch_id, gt_flow in enumerate(batch["radar_flow"]):
                valid_from_pc2res = res_dict['radar_pc0_valid_point_idxes'][batch_id]
                pose_flow = radar_pose_flows[batch_id][valid_from_pc2res]

                final_flow_ = res_dict['radar_flow'][batch_id]
                v1_dict= evaluate_leaderboard(final_flow_, pose_flow, batch['radar_pc0'][batch_id][valid_from_pc2res], gt_flow[valid_from_pc2res], \
                                           batch['radar_flow_is_valid'][batch_id][valid_from_pc2res], batch['radar_flow_category_indices'][batch_id][valid_from_pc2res])
                v2_dict = evaluate_leaderboard_v2(final_flow_, pose_flow, batch['radar_pc0'][batch_id][valid_from_pc2res], gt_flow[valid_from_pc2res], \
                                        batch['radar_flow_is_valid'][batch_id][valid_from_pc2res], batch['radar_flow_category_indices'][batch_id][valid_from_pc2res])
                v3_dict = evaluate_leaderboard_v3(final_flow_, pose_flow, batch['radar_pc0'][batch_id][valid_from_pc2res], gt_flow[valid_from_pc2res], \
                                        batch['radar_flow_is_valid'][batch_id][valid_from_pc2res], batch['radar_flow_category_indices'][batch_id][valid_from_pc2res])

                self.radar_metrics.step(v1_dict, v2_dict, v3_dict)
        else:
            pass

    def on_validation_epoch_end(self):
        self.model.timer.print(random_colors=False, bold=False)
        if self.av2_mode == 'test':
            print(f"\nModel: {self.model.__class__.__name__}, Checkpoint from: {self.load_checkpoint_path}")
            print(f"Test results saved in: {self.save_res_path}, Please run submit to zip the results and upload to online leaderboard.")
            return
        
        if self.av2_mode == 'val':
            print(f"\nModel: {self.model.__class__.__name__}, Checkpoint from: {self.load_checkpoint_path}")
            print(f"More details parameters and training status are in checkpoints")

        self.radar_metrics.normalize()
        for key in self.radar_metrics.bucketed:
            for type_ in 'Static', 'Dynamic':
                self.log(f"Radar val/{type_}/{key}", self.radar_metrics.bucketed[key][type_],sync_dist=True)
        for key in self.radar_metrics.epe_3way:
            self.log(f"Radar val/{key}", self.radar_metrics.epe_3way[key],sync_dist=True)

        self.radar_metrics.print()
        self.radar_metrics = new_OfficialMetrics(class_name= "Radar")

    def eval_only_step_(self, batch, res_dict):
        batch = {key: batch[key][0] for key in batch if len(batch[key])>0}
        res_dict = {key: res_dict[key][0] for key in res_dict if len(res_dict[key])>0}

        if self.vod_mode == "radar_only":
            pose_flow = res_dict['radar_pose_flow']
            final_flow_ = res_dict['radar_flow']

            if 'radar_pc0_valid_point_idxes' in res_dict:
                valid_from_pc2res = res_dict['radar_pc0_valid_point_idxes']
            
            if self.av2_mode == 'val': 
                gt_flow = batch["radar_flow"]

                v1_dict= evaluate_leaderboard(final_flow_, pose_flow[valid_from_pc2res], batch['radar_pc0'][valid_from_pc2res], gt_flow[valid_from_pc2res], \
                                           batch['radar_flow_valid'][valid_from_pc2res], batch['radar_flow_category'][valid_from_pc2res])
                v2_dict = evaluate_leaderboard_v2(final_flow_, pose_flow[valid_from_pc2res], batch['radar_pc0'][valid_from_pc2res], gt_flow[valid_from_pc2res], \
                                           batch['radar_flow_valid'][valid_from_pc2res], batch['radar_flow_category'][valid_from_pc2res])
                v3_dict= evaluate_leaderboard_v3(final_flow_, pose_flow[valid_from_pc2res], batch['radar_pc0'][valid_from_pc2res], gt_flow[valid_from_pc2res], \
                                           batch['radar_flow_valid'][valid_from_pc2res], batch['radar_flow_category'][valid_from_pc2res])
                self.radar_metrics.step(v1_dict, v2_dict, v3_dict)
            
        
    def validation_step(self, batch, batch_idx):
        if self.av2_mode == 'val' or self.av2_mode == 'test':

            if self.vod_mode == "radar_only":
                batch['radar_ft0'] = torch.cat([batch['radar0_vr'], batch['radar0_RCS']], dim=2) # B N 3
                batch['radar_ft1'] = torch.cat([batch['radar1_vr'], batch['radar1_RCS']], dim=2) # B N 3
                npts_1 = batch['radar_ft0'].shape[1]
                npts_2 = batch['radar_ft1'].shape[1]

                if npts_1<self.npoints:
                    sample_idx1 = np.arange(0,npts_1)
                    sample_idx1 = np.append(sample_idx1, np.random.choice(npts_1,self.npoints-npts_1,replace=True))
                else:
                    sample_idx1 = np.random.choice(npts_1, self.npoints, replace=False)

                if npts_2<self.npoints:
                    sample_idx2 = np.arange(0,npts_2)
                    sample_idx2 = np.append(sample_idx2, np.random.choice(npts_2,self.npoints-npts_2,replace=True))
                else:
                    sample_idx2 = np.random.choice(npts_2, self.npoints, replace=False)

                batch['radar_pc0'] = batch['radar_pc0'][:,sample_idx1,:]
                batch['radar_pc1'] = batch['radar_pc1'][:,sample_idx2,:]
                batch['radar_ft0'] = batch['radar_ft0'][:,sample_idx1,:]
                batch['radar_ft1'] = batch['radar_ft1'][:,sample_idx2,:]

                # for evaluation
                batch['radar_flow'] = batch['radar_flow'][:,sample_idx1,:] # B N 3
                batch['radar_flow_valid'] = batch['radar_flow_valid'][:,sample_idx1] # B N
                batch['radar_flow_category'] = batch['radar_flow_category'][:,sample_idx1] # B N
                batch['radar_motion_mask'] = batch['radar_motion_mask'][:,sample_idx1]
                batch['origin_radar_pc0'] = batch['radar_pc0'].clone()

                self.model.timer[12].start("One Scan")

                res_dict = self.model(batch)
                self.model.timer[12].stop()
                self.eval_only_step_(batch, res_dict)
        else:
            res_dict = self.model(batch)
            self.train_validation_step_(batch, res_dict)

    def configure_optimizers(self):
        torch.autograd.set_detect_anomaly(True)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
