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
from src.utils.depthvectorizer import DepthVectorizer
import yaml


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
    parser.add_argument("--vtracer_cfg", default='src/configs/vtracer_default.yml',
                        help="src path with images to process")
    parser.add_argument("--inpainting_type", default='closest',
                        help="'closest' or 'harmonic")
    parser.add_argument("--downsample_factor", default=1,
                        help="'closest' or 'harmonic")

    parser.add_argument("--save_npy", type=bool,
                        default=False, help="Save .npy of predicted depth")
    parser.add_argument("--device", type=str,
                        default='cuda', help="device for the model")
    parser.add_argument("--model_path", type=str,
                        default='checkpoints/mmsvg_model/checkpoints/id_model.ckpt', help="device for the model")
    args = parser.parse_args()

    to_process = find_images(args.src)
    if len(to_process) == 0:
        print("No images found in {}".format(args.src))
        exit(0)

    print("Found {} images to process\n".format(len(to_process)))
    print("Loading Illustrator's Depth model...")
    model, cfg = load_illustrators_depth_model(
        args.model_path, device=args.device)
    print("\nPredicting Illustrator's Depth...")

    output_dir = args.src+'_vectorization'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.vtracer_cfg, 'r') as f:
        vtracer_cfg = yaml.safe_load(f)
    V = DepthVectorizer(vtracer_cfg=vtracer_cfg,
                        downsample_factor=args.downsample_factor, inpainting_type=args.inpainting_type)

    for path in tqdm(to_process):
        raw_name = ''.join(path.split('.')[:-1]).split('/')[-1]

        V.predict_and_vectorize(
            path, model, output_path=f'{output_dir}/{raw_name}')

        if args.save_npy:
            np.save(f'{output_dir}/{raw_name}.npy', V.predicted_depth)
