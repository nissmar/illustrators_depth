"""
ADOBE

Copyright 2026 Adobe

All Rights Reserved.

NOTICE: All information contained herein is, and remains
the property of Adobe and its suppliers, if any. The intellectual
and technical concepts contained herein are proprietary to Adobe
and its suppliers and are protected by all applicable intellectual
property laws, including trade secret and copyright laws.
Dissemination of this information or reproduction of this material
is strictly forbidden unless prior written permission is obtained
from Adobe.
"""

from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning import loggers
import yaml
import lightning as pL
from src.data.dataset import DepthData, custom_transforms
from src.model.illustrators_depth_model import IllustratorsDepthModel, load_illustrators_depth_model
import torch
import argparse
import depth_pro
import torchvision.transforms as T
from datasets import load_dataset


torch.set_float32_matmul_precision("high")


class TBLogger(loggers.TensorBoardLogger):
    @rank_zero_only
    def log_metrics(self, metrics, step):
        metrics.pop("epoch", None)
        return super().log_metrics(metrics, step)


if __name__ == "__main__":
    pL.seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str,
                        default="src/configs/cnn_decoder.yml")
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
    data = load_dataset("OmniSVG/MMSVG-Illustration", cache_dir="./")["train"]

    if cfg["training"]["load_from_checkpoint"]:
        dpcfg = depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT
        dpcfg.checkpoint_uri = None
        model = IllustratorsDepthModel.load_from_checkpoint(
            cfg["training"]["load_from_checkpoint"], cfg=cfg, dpcfg=dpcfg)
        model.train()
    else:
        dpcfg = depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT
        dpcfg.checkpoint_uri = "./ml-depth-pro/checkpoints/depth_pro.pt"
        model = IllustratorsDepthModel(cfg, dpcfg)

    train_data = DepthData(
        data,
        input_size=cfg["data"]["input_size"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        shuffle=True,
        idx_files=cfg["data"]["train_idx_files"],
        raster_resolution=cfg["data"]["raster_resolution"],
        transforms=custom_transforms(cfg["data"]["input_size"], cfg["data"]["train_transform"], mean=[
                                     0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        concat_n=cfg["data"]["concat_n"] if "concat_n" in cfg["data"] else None,
        squash_consecutive_same_color=cfg["data"]["squash_consecutive_same_color"],
    )

    val_data = DepthData(
        data,
        input_size=cfg["data"]["input_size"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        min_files=cfg["data"]["val_min_files"],
        max_files=cfg["data"]["val_max_files"],
        shuffle=False,
        raster_resolution=cfg["data"]["raster_resolution"],
        transforms=custom_transforms(cfg["data"]["input_size"], None, mean=[
                                     0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        concat_n=cfg["data"]["concat_n"] if "concat_n" in cfg["data"] else None,
        squash_consecutive_same_color=cfg["data"]["squash_consecutive_same_color"],
    )

    checkpoint_callback = pL.pytorch.callbacks.ModelCheckpoint(
        save_top_k=-1,  # save all epochs
        save_on_train_epoch_end=True,  # all epochs
        save_weights_only=True,
    )
    lr_monitor = pL.pytorch.callbacks.LearningRateMonitor(
        logging_interval='step')
    trainer = pL.Trainer(
        max_epochs=cfg["training"]["max_epochs"],
        val_check_interval=cfg["training"]["val_check_interval"],
        log_every_n_steps=cfg["training"]["log_every_n_steps"],
        logger=TBLogger(
            cfg["logger"]["save_dir"],
            name=cfg["logger"]["name"],
            default_hp_metric=False,
        ),
        num_nodes=2,
        devices=cfg["training"]["ndevices"] if "ndevices" in cfg["training"] else 2,
        strategy="ddp_find_unused_parameters_true",
        gradient_clip_val=cfg["training"]["gradient_clip_val"],
    )
    trainer.callbacks[-1].save_weights_only = True
    trainer.callbacks.append(checkpoint_callback)
    trainer.callbacks.append(lr_monitor)
    trainer.fit(model, train_data.dataloader(), val_data.dataloader())
