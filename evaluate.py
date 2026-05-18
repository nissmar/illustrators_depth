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
import argparse
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.eval.metrics import order_metric, mse_median, mae_median, calc_lpips, calc_ssim, calc_mae, calc_mse, calc_path_number
import torch


def find_images(src_path):
    to_process = glob.glob(f"{src_path}/*_depth.npy", recursive=False)
    return to_process


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rasterize all SVG in a directory"
    )
    parser.add_argument("--gt_src", help="src path")
    parser.add_argument("--pred_src", help="prediction path")
    parser.add_argument("--eval_SVG", type=bool,
                        default=False, help="Eval SVG or just depth")
    args = parser.parse_args()

    to_process_gt = find_images(args.gt_src)
    names = [e.split('/')[-1] for e in to_process_gt]
    if len(names) == 0:
        print("No images found in {}".format(args.gt_src))
        exit(0)

    lpred = len(find_images(args.pred_src))
    if lpred != len(names):
        print(
            f"ERROR: found {lpred} predicted depths and {len(names)} gt depths")

    print("Found {} images to process\n".format(len(names)))

    if args.eval_SVG == False:
        print('Evaluating predicted depth only')
        losses = {'si_mae': [], 'si_mse': [], 'order_loss': []}
    else:
        print('Evaluating SVGs')
        losses = {'mae_depth': [], 'mse_depth': [], 'order_loss': [],
                  'l2_img': [], 'ssim': [], 'path_number': [], 'lpips': []}

    for name in tqdm(names):
        gt_depth = np.load(f'{args.gt_src}/{name}')
        pred_depth = np.load(f'{args.pred_src}/{name}')

        if args.eval_SVG == False:
            pred_depth = torch.tensor(pred_depth).unsqueeze(0)
            gt_depth = torch.tensor(gt_depth).unsqueeze(0)
            losses['si_mae'].append(mae_median(pred_depth, gt_depth))
            losses['si_mse'].append(mse_median(pred_depth, gt_depth))
            losses['order_loss'].append(order_metric(pred_depth, gt_depth))
        else:
            name_im = name.replace('_depth.npy', '.png')
            gt_img = plt.imread(f'{args.gt_src}/{name_im}')[..., :3]
            rec_img = plt.imread(f'{args.pred_src}/{name_im}')[..., :3]

            losses['mae_depth'].append(calc_mae(pred_depth, gt_depth))
            losses['mse_depth'].append(calc_mse(pred_depth, gt_depth))
            losses['order_loss'].append(order_metric(torch.tensor(
                pred_depth).unsqueeze(0), torch.tensor(gt_depth).unsqueeze(0)))
            losses['path_number'].append(
                calc_path_number(pred_depth, gt_depth))

            losses['lpips'].append(calc_lpips(rec_img, gt_img))
            losses['ssim'].append(calc_ssim(rec_img, gt_img))
            losses['l2_img'].append(calc_mse(rec_img, gt_img))

    if args.eval_SVG == False:
        print(
            f"SRC      & Order & MAE & MSE ")
        print(
            f"{args.pred_src} & {np.mean(losses['order_loss']):.3f} & {np.mean(losses['si_mae']):.2f} & {np.mean(losses['si_mse']):.2f} ")
    else:
        print(
            f"SRC      & Order & MAE & MSE & Path Number & MSE (x1e-2) & SSIM & LPIPS  ")
        print(
            f"{args.pred_src}  & {np.mean(losses['order_loss']):.3f} & {np.mean(losses['mae_depth']):.2f} & {np.mean(losses['mse_depth']):.2f}  & {np.mean(losses['path_number']):.2f} & {100*np.mean(losses['l2_img']):.3f} & {np.mean(losses['ssim']):.3f} & {np.mean(losses['lpips']):.3f}")
