"""
# Created: 2023-07-12 19:30
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Description: Train Model
"""

import torch
torch.autograd.set_detect_anomaly(True)
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint
)

from omegaconf import DictConfig, OmegaConf
import hydra, wandb, os, time
# from hydra.core.hydra_config import HydraConfig
from pathlib import Path
from scripts.network.dataloader import radarflow_VODH5Dataset,  radarflow_vod_collate_fn_pad
from scripts.pl_model import ModelWrapper 

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# select which model to run
# by change config_name 
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    pl.seed_everything(cfg.seed, workers=True)
    output_dir = cfg.my_model_save_path + cfg.model_name + '/'
    t = time.localtime()
    timesave = str(t.tm_year) + str("%02d"%t.tm_mon) + str("%02d"%t.tm_mday) + str("%02d"%t.tm_hour) + str("%02d"%t.tm_min) # str(t.tm_sec)

    if cfg.dataset_name == "VOD":
        if cfg.model_name == "IterFlow" :
            train_dataset = radarflow_VODH5Dataset(cfg.dataset_path, "/train") 
            train_loader = DataLoader(train_dataset,
                                    batch_size=cfg.batch_size,
                                    shuffle=True,
                                    num_workers=cfg.num_workers,
                                    collate_fn= radarflow_vod_collate_fn_pad , 
                                    pin_memory=True)
            
            val_loader = DataLoader(radarflow_VODH5Dataset(cfg.dataset_path, "/val"),
                                    batch_size=cfg.batch_size,
                                    shuffle=False,
                                    num_workers=cfg.num_workers,
                                    collate_fn= radarflow_vod_collate_fn_pad ,  
                                    pin_memory=True)

    

    model_name = cfg.model_name
    os.makedirs(output_dir, exist_ok=True)
    my_save_path = os.path.join(output_dir,  "checkpoints" + timesave)
    Path(my_save_path).mkdir(parents=True, exist_ok=True)
    
    cfg = DictConfig(OmegaConf.to_container(cfg, resolve=True))
    if cfg.dataset_name in [ "VOD"]:
        if cfg.model_name == "IterFlow":   
            model = ModelWrapper(cfg)  
        else:
            raise ValueError("Unavailable cfg.model_name")


    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"+ timesave),
            filename="{epoch:02d}_"+model_name,
            auto_insert_metric_name=False,
            monitor=cfg.model.val_monitor,
            mode="min",
            save_top_k=cfg.save_top_model
        ),
        LearningRateMonitor(logging_interval="epoch")
    ]

    wandb_logger = WandbLogger(save_dir=output_dir,
                               entity="SF",
                               project=f"{cfg.wandb_project_name}", 
                               name=f"{cfg.output}",
                               offline=(cfg.wandb_mode == "offline"),
                               log_model=(True if cfg.wandb_mode == "online" else False))
    
    if len(cfg.available_gpus) > 1:
        trainer = pl.Trainer(logger=wandb_logger,                            
                                log_every_n_steps=50,
                                accelerator="gpu",
                                devices=cfg.available_gpus,
                                check_val_every_n_epoch=cfg.val_every,
                                gradient_clip_val=cfg.gradient_clip_val,
                                # strategy="ddp_find_unused_parameters_false" , # if cfg.gpus > 1 else "bagua",# "auto"

                                # MY:        
                                strategy="ddp" ,
                                # strategy="ddp_find_unused_parameters_false" ,
                                callbacks=callbacks,
                                max_epochs=cfg.epochs,
                                sync_batchnorm=cfg.sync_bn)
    else:
        trainer = pl.Trainer(logger=wandb_logger,
                                log_every_n_steps=50,
                                accelerator="gpu",
                                devices=cfg.available_gpus,
                                check_val_every_n_epoch=cfg.val_every,
                                gradient_clip_val=cfg.gradient_clip_val,
            
                                callbacks=callbacks,
                                max_epochs=cfg.epochs,
                                sync_batchnorm=cfg.sync_bn)
    
    wandb_logger.watch(model, log_graph=False)

    if trainer.global_rank == 0:
        print("\n"+"-"*40)
        print("Initiating wandb and trainer successfully.  ^V^ ")
        print(f"We will use {cfg.gpus} GPUs to train the model. Check the checkpoints in {output_dir} checkpoints folder.")
        print("Total Train Dataset Size: ", len(train_dataset))
        print("-"*40+"\n")

    # NOTE(Qingwen): search & check: def training_step(self, batch, batch_idx)
    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = val_loader, ckpt_path = cfg.checkpoint)
    wandb.finish()

if __name__ == "__main__":
    main()