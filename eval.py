import torch
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelSummary
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
import hydra, wandb, os, sys
from hydra.core.hydra_config import HydraConfig
from scripts.network.dataloader import radarflow_VODH5Dataset,  radarflow_vod_collate_fn_pad
from scripts.pl_model import ModelWrapper 

from pathlib import Path

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


@hydra.main(version_base=None, config_path="conf", config_name="eval") # cylinder_eval my_eval
def main(cfg):
    pl.seed_everything(cfg.seed, workers=True)
    # TO DO output
    output_dir = cfg.save_path + cfg.model_name + "-eval/" 
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(cfg.checkpoint):
        print(f"Checkpoint {cfg.checkpoint} does not exist. Need checkpoints for evaluation.")
        sys.exit(1)
    

    cfg.output = cfg.model_name + f"-{cfg.av2_mode}"

    if cfg.dataset_name in ["VOD"]:
        if  cfg.model_name == "IterFlow":              
            mymodel = ModelWrapper.load_from_checkpoint(cfg.checkpoint, cfg=cfg, eval=True)
        else:
            raise ValueError("Unavailable cfg.model_name")


    wandb_logger = WandbLogger(save_dir=output_dir,
                               entity="kth-rpl",
                               project= cfg.model_name + "-eval", 
                               name=f"{cfg.output}",
                               offline=(cfg.wandb_mode == "offline"))
    
    trainer = pl.Trainer(logger=wandb_logger, accelerator='gpu',devices=cfg.available_gpus)

    if cfg.dataset_name == "VOD":
        if cfg.model_name == "IterFlow":
            trainer.validate(model = mymodel, \
                        dataloaders = DataLoader(radarflow_VODH5Dataset(cfg.dataset_path, cfg.cmflow_dataset_path, "/val"), batch_size=1, num_workers=cfg.num_workers, shuffle=False))

    wandb.finish()

if __name__ == "__main__":
    main()