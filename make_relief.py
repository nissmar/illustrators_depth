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
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import numpy as np
import argparse


def flat_surface_mesh(pred_depth):
    """
    Generate a (V, F) mesh for a flat surface of size NxN.
    V: (N*N, 3) array of vertex positions
    F: (2*(N-1)*(N-1), 3) array of face indices (triangles)
    """
    # Vertices
    N, M = pred_depth.shape
    xv, yv = np.meshgrid(np.arange(N), np.arange(M), indexing='ij')
    V = np.stack([yv.ravel()/N, xv.ravel()/N, pred_depth.ravel()], axis=1)

    # Faces
    F = []
    for i in range(N-1):
        for j in range(M-1):
            idx = i*M + j
            # two triangles per quad
            F.append([idx+1, idx, idx+M])
            F.append([idx+M, idx+M+1, idx+1])
    F = np.array(F)
    return V, F


def export_mesh_to_obj(filename, V, F):
    """
    Export mesh vertices (V) and faces (F) to a Wavefront OBJ file.
    """
    with open(filename, 'w') as f:
        for v in V:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in F:
            # OBJ format uses 1-based indexing
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def pad_image(img, pad_width, pad_value=0):
    """
    Pad an image (numpy array) with specified width and value.
    img: np.ndarray, shape (H, W, C) or (H, W)
    pad_width: int or tuple, number of pixels to pad on each side
    pad_value: int or tuple, value to pad with
    """
    if img.ndim == 3:
        pad_width_tuple = ((pad_width, pad_width),
                           (pad_width, pad_width), (0, 0))
    else:
        pad_width_tuple = ((pad_width, pad_width), (pad_width, pad_width))
    return np.pad(img, pad_width_tuple, mode='constant', constant_values=pad_value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make a relief mesh from illustrator's depth"
    )
    parser.add_argument("--input_depth", type=str,
                        help="input illustrator's depth")
    parser.add_argument("--downsample", default=4,
                        type=int, help="Downsampling factor")
    parser.add_argument("--blur_sigma", default=2.0,
                        type=float, help="Blur sigma for Gaussian filter")
    parser.add_argument("--height", default=30, type=int,
                        help="Height scale for the relief mesh")
    parser.add_argument("--output", default="output.obj", type=str,
                        help="Output OBJ file path")
    args = parser.parse_args()
    pred_depth = plt.imread(args.input_depth).sum(-1)

    blurred_depth = gaussian_filter(
        pred_depth[::args.downsample, ::args.downsample], sigma=args.blur_sigma)
    blurred_depth = 1-(blurred_depth - blurred_depth.min()) / \
        (blurred_depth.max() - blurred_depth.min())
    V, F = flat_surface_mesh(blurred_depth/args.height)
    export_mesh_to_obj(args.output, V, F)
