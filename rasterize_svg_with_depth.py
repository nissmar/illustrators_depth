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
from src.data.data_utils import svg_to_depth_svg, rasterize_depth_svg, rasterize_svg


def find_images(src_path):
    to_process = glob.glob(f"{src_path}/*.svg", recursive=False)
    return to_process


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rasterize all SVG in a directory"
    )
    parser.add_argument("--src", help="src path with images to process")
    parser.add_argument("--save_npy", type=bool,
                        default=True, help="Save .npy of rasterized depth")
    parser.add_argument("--resolution", default=1536)
    parser.add_argument("--squash_same_color", default=True)
    parser.add_argument("--background_color", default='white')
    args = parser.parse_args()

    to_process = find_images(args.src)
    if len(to_process) == 0:
        print("No images found in {}".format(args.src))
        exit(0)

    print("Found {} images to process\n".format(len(to_process)))

    for path in tqdm(to_process):
        with open(path, 'r') as f:
            svg_file = f.read()
        depth_svg_file = svg_to_depth_svg(
            svg_file, make_crisp=True, squash_consecutive_same_color=args.squash_same_color)  # crisp by default
        # rgb_image
        rgb_img = rasterize_svg(
            svg_file, resolution=args.resolution, background_color=args.background_color)
        # depth_image
        depth_img = rasterize_depth_svg(
            depth_svg_file, resolution=args.resolution)
        _, ngt = np.unique(depth_img, return_inverse=True)
        depth_img = ngt.reshape(depth_img.shape)
        save_name = path.replace('.svg', f'.png')
        plt.imsave(save_name, np.array(rgb_img))
        plt.imsave(save_name.replace('.png', '_depth.png'),
                   depth_img, cmap='inferno')
        if args.save_npy:
            np.save(save_name.replace('.png', '_depth.npy'), depth_img)
