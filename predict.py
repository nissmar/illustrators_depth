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

import os
import numpy as np
import argparse
import glob
from tqdm import tqdm
from src.model.illustrators_depth_model import load_illustrators_depth_model
import matplotlib.pyplot as plt


def find_images(src_path, exlude_depth_preds=True):
    to_process = glob.glob(f"{src_path}/*.png", recursive=False)+glob.glob(
        f"{src_path}/*.jpeg", recursive=False)+glob.glob(f"{src_path}/*.jpg", recursive=False)
    if exlude_depth_preds:
        to_process = [e for e in to_process if e.split(
            '.')[-2][-6:] != '_depth']
    return to_process


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict illustrator's depth of all images in a directory"
    )
    parser.add_argument("--src", help="src path with images to process")
    parser.add_argument("--gaussian_sigma", type=float, default=0,
                        help="Standard deviation for Gaussian smoothing applied to the input image before inference. Default is 0 (no smoothing).")
    parser.add_argument("--preserve_size", type=bool, default=True,
                        help="If False, image will be rescaled to 1536x1536")
    parser.add_argument("--save_npy", type=bool,
                        default=True, help="Save .npy of predicted depth")
    parser.add_argument("--device", type=str,
                        default='cuda', help="device for the model")
    parser.add_argument("--model_path", type=str,
                        default='checkpoints/mmsvg_model/checkpoints/id_model.ckpt', help="device for the model")
    parser.add_argument("--clip_percentile", type=float,
                        default=0, help="clips percentile")
    parser.add_argument("--exclude_depth_preds", type=bool,
                        default=True, help="exclude all files ending with '_depth'")
    args = parser.parse_args()

    to_process = find_images(args.src)
    print("\n")
    if len(to_process) == 0:
        print("No images found in {}".format(args.src))
        exit(0)

    print("Found {} images to process\n".format(len(to_process)))
    print("Loading Illustrator's Depth model...")
    model, cfg = load_illustrators_depth_model(
        args.model_path, device=args.device)
    print("\nPredicting Illustrator's Depth...")

    output_dir = args.src+'_predictions'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"\nTarget path: {output_dir}")
    for path in tqdm(to_process):
        np_img, pred_depth = model.infer_single_image(
            path, preserve_size=args.preserve_size, gaussian_sigma=args.gaussian_sigma)
        raw_name = ''.join(path.split('.')[:-1]).split('/')[-1]
        if args.save_npy:
            np.save(f'{output_dir}/{raw_name}_depth.npy', pred_depth)
        if args.clip_percentile > 0:
            eps = args.clip_percentile
            vmin, vmax = np.quantile(pred_depth, [eps, 1-eps])
        else:
            vmin = pred_depth.min()
            vmax = pred_depth.max()
        plt.imsave(f'{output_dir}/{raw_name}_depth.png', pred_depth,
                   vmin=vmin, vmax=vmax, cmap='inferno')
