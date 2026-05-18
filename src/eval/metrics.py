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

import numpy as np
from src.train.loss import reg_median, sample_points
import torch
from skimage.metrics import structural_similarity as ssim
import lpips
from torchvision import transforms


def order_metric(pred_depth, gt_depth, n_points=50000):
    """Sample n_points from the image and compute the loss between the difference of the predicted and ground truth depth at the sampled points"""
    pts1 = sample_points(gt_depth.shape[-1], n_points, 'cpu')
    pts2 = sample_points(gt_depth.shape[-1], n_points, 'cpu')
    diff_1 = torch.sign(pred_depth[..., pts1[0], pts1[1]] -
                        pred_depth[..., pts2[0], pts2[1]])
    diff_gt = torch.sign(
        gt_depth[..., pts1[0], pts1[1]] - gt_depth[..., pts2[0], pts2[1]])
    return ((diff_gt[diff_gt != 0]*diff_1[diff_gt != 0] >= 0).float().mean()).item()


def mse_median(pred_depth, gt_depth):
    inverse_gt = reg_median(gt_depth.to(pred_depth.dtype))
    pred = reg_median(pred_depth)
    return ((pred-inverse_gt)**2).mean()


def mae_median(pred_depth, gt_depth):
    inverse_gt = reg_median(gt_depth.to(pred_depth.dtype))
    pred = reg_median(pred_depth)
    return (pred-inverse_gt).abs().mean()


loss_fn_lpips = lpips.LPIPS(net='alex')
transform_lpips = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def calc_lpips(pred, gt):
    lpips_score = loss_fn_lpips(transform_lpips(gt).unsqueeze(
        0), transform_lpips(pred).unsqueeze(0))
    return lpips_score.item()


def calc_ssim(pred, gt):
    return ssim(gt, pred, multichannel=True, data_range=1, channel_axis=-1)


def calc_mae(pred, gt):
    return np.abs(gt - pred).mean()


def calc_mse(pred, gt):
    return ((gt - pred)**2).mean()


def calc_path_number(pred_depth, gt_depth):
    return np.abs(pred_depth.max()-gt_depth.max())/gt_depth.max()
