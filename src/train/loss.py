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
import torch.nn as nn


def sample_points(max_coord, num_points, device):
    pts = torch.randint(0, max_coord, (2, num_points), device=device)
    return pts


glob_loss = nn.BCELoss()


def loss_order(diff_pred, diff_gt, fac):
    """Compute the loss between the difference of the predicted and ground truth depth"""
    return glob_loss(torch.sigmoid(diff_pred * fac), torch.sigmoid(diff_gt))


def loss_sample(out, gt_depth, n_points=1000, fac=10):
    """Sample n_points from the image and compute the loss between the difference of the predicted and ground truth depth at the sampled points"""
    pts1 = sample_points(gt_depth.shape[-1], n_points, gt_depth.device)
    pts2 = sample_points(gt_depth.shape[-1], n_points, gt_depth.device)

    diff_1 = out[..., pts1[0], pts1[1]] - out[..., pts2[0], pts2[1]]
    diff_gt = gt_depth[..., pts1[0], pts1[1]] - gt_depth[..., pts2[0], pts2[1]]
    return loss_order(diff_1, diff_gt, fac)


def roll_diff(tensor):
    """Compute finite difference using torch.roll along axes"""
    # dx: difference along width (axis=-1)
    dx = tensor - torch.roll(tensor, shifts=1, dims=-1)
    dy = tensor - torch.roll(tensor, shifts=1, dims=-2)
    return dx, dy


def loss_grad(out, gt_depth, fac=10):
    """Compute the gradient of the predicted and ground truth depth"""
    g_out = torch.cat(roll_diff(out), dim=0)
    g_gt = torch.cat(roll_diff(gt_depth), dim=0)
    return loss_order(g_out, g_gt, fac)


def loss_l1_grad(out, gt_depth, eps=1e-10):
    """Compute the gradient of the predicted and ground truth depth"""
    g_out = torch.cat(roll_diff(out), dim=0)
    g_gt = torch.cat(roll_diff(gt_depth), dim=0)
    return g_out[g_gt < eps].abs().mean()


def loss_region(pred_depth, gt_depth):
    loss = 0
    for i in range(pred_depth.shape[0]):
        k = gt_depth[i].flatten()[torch.randint(
            0, gt_depth[i].numel(), (1,)).item()]
        region = pred_depth[i, gt_depth[i] == k]
        loss += (region - region.mean().detach()).pow(2).sum() / pred_depth.shape[
            -1
        ] ** 2
    return loss


def loss_region_sample(pred_depth, gt_depth, fac=10):
    mean_loss = 0
    diff_loss = 0
    for i in range(pred_depth.shape[0]):
        unique_depths = torch.unique(gt_depth[i])
        idxs = unique_depths[torch.randperm(len(unique_depths))[
            :2]].sort().values
        r1 = pred_depth[i, gt_depth[i] == idxs[0]]
        r2 = pred_depth[i, gt_depth[i] == idxs[1]]

        m_r1 = r1.mean()
        m_r2 = r2.mean()

        mean_loss += (r1 - m_r1.detach()).pow(2).mean() + (r2 - m_r2.detach()).pow(
            2
        ).mean()
        diff_loss += loss_order(m_r2 - m_r1, idxs[1] - idxs[0], fac)
    return mean_loss, diff_loss


def reg_median(depth):
    bs = depth.shape[0]
    f_depth = depth.view(bs, -1)
    t = f_depth.median(-1, keepdim=True)[0]
    s = (f_depth-t).abs().mean(-1, keepdim=True)
    return (f_depth-t)/torch.clamp(s, min=1e-10)


def loss_inverse_median(pred_depth, gt_depth):
    inverse_gt = reg_median(gt_depth.to(pred_depth.dtype))
    pred = reg_median(pred_depth)
    return (pred-inverse_gt).abs().mean()


def loss_l1_grad_median(out, gt_depth):
    """Compute the gradient of the predicted and ground truth depth"""
    s = gt_depth.shape[-1]
    inverse_gt = reg_median(gt_depth.to(out.dtype)).view(-1, s, s)
    pred = reg_median(out).view(-1, s, s)
    g_out = torch.cat(roll_diff(pred), dim=0)
    g_gt = torch.cat(roll_diff(inverse_gt), dim=0)
    return (g_out-g_gt).abs().mean()


def loss_inverse_median2(pred_depth, gt_depth):
    # inv_out = -1.0 / torch.clamp(pred_depth, min=1e-4, max=1e4)
    inv_gt = 1.0 / torch.clamp(gt_depth.max()+1-gt_depth, min=1e-4, max=1e4)
    inverse_gt = reg_median(inv_gt.to(pred_depth.dtype))
    pred = reg_median(pred_depth)
    return (pred-inverse_gt).abs().mean()
