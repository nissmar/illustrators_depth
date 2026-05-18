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

import torch
import torchvision
import lightning as pL
from src.utils.prompt_utils import resolve_path
from src.data.dataset import load_single_image, tensor_to_numpy_rgb, process_pil
import numpy as np
import depth_pro
import yaml
from typing import Union
import skimage
import PIL


class IllustratorsDepthModel(pL.LightningModule):
    def __init__(self, cfg, dpcfg):
        super().__init__()
        self.img_size = cfg["data"]["input_size"]
        self.loss_dict = cfg["training"]["loss_dict"]
        self.lr = cfg["training"]["lr"]
        dp_model, transform = depth_pro.create_model_and_transforms(
            config=dpcfg)
        dp_model.fov = None
        dpcfg.checkpoint_uri = None
        self.encoder = dp_model.encoder
        self.decoder = dp_model.decoder
        self.head = dp_model.head
        self.save_hyperparameters(ignore=["dpcfg"])

    def init_encoder_decoder_head(self, encoder, decoder, head):
        self.encoder = encoder
        self.decoder = decoder
        self.head = head

    def forward(self, x):
        encodings = self.encoder(x)
        features, _ = self.decoder(encodings)
        inverse_depth = self.head(features)
        # this is legacy scaling, it doesn't change anything to train using inverse_depth
        pred_depth = -1.0 / torch.clamp(inverse_depth, min=1e-4, max=1e4)
        return pred_depth

    def compute_losses(self, pred_depth, gt_depth, raw_rgb):
        losses = {}
        for l in self.loss_dict:
            loss_fn = resolve_path(l)
            weight = self.loss_dict[l]["weight"]
            if 'kwargs' in self.loss_dict[l]:
                kwargs = self.loss_dict[l]["kwargs"]
                loss_res = loss_fn(pred_depth.squeeze(1), gt_depth, **kwargs)
            else:
                loss_res = loss_fn(pred_depth.squeeze(1), gt_depth)
            losses[l] = weight * loss_res
        return sum([losses[k] for k in losses]), losses

    def training_step(self, batch):
        raw_rgb, gt_depth = batch
        pred_depth = self(raw_rgb)

        loss, losses = self.compute_losses(pred_depth, gt_depth, raw_rgb)
        self.log("loss", loss)
        for k in losses:
            self.log(k, losses[k])
        sch = self.lr_schedulers()
        sch.step()
        return loss

    def log_pred_img(self, pred_depth):
        if pred_depth.shape[1] == 1:
            pred_depth = pred_depth.squeeze(1)
            flat_pred = pred_depth.view(pred_depth.shape[0], -1)
            pred_img = (pred_depth - flat_pred.min(1).values[..., None, None]) / (
                flat_pred.max(1).values - flat_pred.min(1).values
            )[..., None, None]
            pred_img = pred_img.permute(1, 0, 2).reshape(pred_img.shape[1], -1)
            pred_img = (pred_img * 255).cpu().detach().numpy().astype(np.uint8)
            self.logger.experiment.add_image(
                "val_pred_depth_img", pred_img, self.global_step, dataformats="HW"
            )

    def validation_step(self, batch, batch_idx):
        raw_rgb, gt_depth = batch
        pred_depth = self(raw_rgb)

        loss, losses = self.compute_losses(pred_depth, gt_depth, raw_rgb)

        self.log("val_loss", loss, sync_dist=True)
        for k in losses:
            self.log("val_" + k, losses[k], sync_dist=True)

        if batch_idx == 0:
            self.log_pred_img(pred_depth[:5])
        return pred_depth, gt_depth

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {"params": self.encoder.parameters(), "lr": self.lr["encoder"]},
            {"params": self.decoder.parameters(), "lr": self.lr["decoder"]},
            {"params": self.head.parameters(), "lr": self.lr["head"]},
        ])
        scheduler = resolve_path(self.lr["scheduler"]["target"])
        scheduler = scheduler(optimizer, **self.lr["scheduler"]["kwargs"])
        return [optimizer], [scheduler]

    @torch.no_grad()
    def infer_single_image(self, image: Union[str, torch.Tensor, PIL.Image.Image], preserve_size=False, gaussian_sigma=0):
        """
        Infer depth from an image using the given model.

        Args:
            model: The depth model (IllustratorsDepthModel).
            image: Either a file path (str) to an image or a torch.Tensor of shape [C, H, W] or [1, C, H, W].
            preserve_size: If False, image will be rescaled to 1536x1536.
            gaussian_sigma: Standard deviation for Gaussian smoothing applied to the input image before inference. Default is 0 (no smoothing).

        Returns:
            np_img: The numpy RGB image (for visualization).
            pred_depth: The predicted depth map as a numpy array.
        """
        self.eval()
        if isinstance(image, str):
            if preserve_size:
                tensor_img, np_img, orishape = load_single_image(
                    image, return_size=preserve_size)
            else:
                tensor_img, np_img = load_single_image(
                    image, return_size=False)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                tensor_img = image.unsqueeze(0)
            elif image.dim() == 4:
                tensor_img = image
            else:
                raise ValueError(
                    "Input tensor must have 3 or 4 dimensions (C,H,W) or (B,C,H,W)")
            np_img = tensor_to_numpy_rgb(tensor_img[0])
        elif isinstance(image, PIL.Image.Image):
            if preserve_size:
                tensor_img, np_img, orishape = process_pil(
                    image, return_size=preserve_size)
            else:
                tensor_img, np_img = process_pil(
                    image, return_size=False)
        tensor_img = tensor_img.to(self.device)
        if gaussian_sigma > 0:
            gaussian_sigma = torchvision.transforms.GaussianBlur(
                kernel_size=13, sigma=gaussian_sigma)
            tensor_img = gaussian_sigma(tensor_img)
        pred_depth = self(tensor_img)
        pred_depth = pred_depth.squeeze().cpu().numpy()
        if preserve_size:
            H, W = orishape
            if H > W:
                H = int(H/W*1536)
                W = 1536

            else:
                W = int(W/H*1536)
                H = 1536
            pred_depth = skimage.transform.resize(
                pred_depth, (W, H), preserve_range=True, anti_aliasing=True)
            np_img = skimage.transform.resize(
                np_img, (W, H), preserve_range=True, anti_aliasing=True)
        return np_img, pred_depth


def load_illustrators_depth_model(model_path, device='cuda') -> IllustratorsDepthModel:
    '''loads an illustrator's depth model from a checkpoint path'''
    with open('/'.join(model_path.split('/')[:-2])+"/hparams.yaml", "r") as f:
        r_cfg = yaml.safe_load(f)
        cfg = r_cfg['cfg']
    dpcfg = depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT
    dpcfg.checkpoint_uri = None
    model = IllustratorsDepthModel.load_from_checkpoint(
        model_path, cfg=cfg, dpcfg=dpcfg, map_location=device)
    model.eval()
    print('Model loaded in eval mode.')
    return model, cfg
