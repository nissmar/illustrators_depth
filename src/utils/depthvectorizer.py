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

from src.utils.svg_tracing_utils import vtracer_trace, average_depth_on_regions, simplify_depth, extract_and_inpaint_layers, create_svg
from src.data.data_utils import svg_to_depth_svg, rasterize_depth_svg, rasterize_svg
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import yaml

with open('src/configs/vtracer_default.yml', 'r') as f:
    DEFAULT_CLUSTER_CFG = yaml.safe_load(f)


class DepthVectorizer:
    def __init__(self, vtracer_cfg=DEFAULT_CLUSTER_CFG, downsample_factor=1, inpainting_type='closest', color_dist_threshold=0.05, resolution=1536, potrace_blur=0, potrace_opttolerance=0.2):
        self.vtracer_cfg = vtracer_cfg
        self.downsample_factor = downsample_factor
        self.inpainting_type = inpainting_type
        self.resolution = resolution
        self.color_dist_threshold = color_dist_threshold
        self.potrace_blur = potrace_blur
        self.potrace_opttolerance = potrace_opttolerance

    def get_clusters(self, input_path):
        # img: H, W, 3, [0, 1]
        cf = self.vtracer_cfg
        svg_str = vtracer_trace(input_path, save=False,
                                colormode=cf['colormode'],
                                hierarchical=cf['hierarchical'],
                                mode=cf['mode'],
                                filter_speckle=cf['filter_speckle'],
                                color_precision=cf['color_precision'],
                                layer_difference=cf['layer_difference'],
                                corner_threshold=cf['corner_threshold'],
                                length_threshold=cf['length_threshold'],
                                max_iterations=cf['max_iterations'],
                                splice_threshold=cf['splice_threshold'],
                                path_precision=cf['path_precision']
                                )
        vtracer_depth = rasterize_depth_svg(
            svg_to_depth_svg(svg_str), self.resolution)
        raster = rasterize_svg(svg_str, self.resolution)
        return vtracer_depth, raster

    def predict_and_vectorize(self, input_path: str, model: nn.Module, output_path: str = 'tmp/result.svg'):
        np_img, predicted_depth = model.infer_single_image(input_path)
        self._vectorize(input_path, np_img, predicted_depth, output_path)

    def vectorize_path(self, input_path: str, depth_path: str, output_path: str = 'tmp/result.svg'):
        np_img = plt.imread(input_path)[..., :3]
        predicted_depth = np.load(depth_path)
        self._vectorize(input_path, np_img, predicted_depth, output_path)

    def _vectorize(self, input_path: str, np_img: np.array, predicted_depth: np.array, output_path: str = 'tmp/result.svg'):
        self.predicted_depth = predicted_depth
        vtracer_depth, raster = self.get_clusters(input_path)
        average_depth = average_depth_on_regions(
            vtracer_depth, predicted_depth)
        simplified_depth, average_colors = simplify_depth(
            average_depth, np_img, self.color_dist_threshold)
        self.average_colors = average_colors
        self.simplified_depth = simplified_depth
        layers = extract_and_inpaint_layers(
            simplified_depth[::self.downsample_factor, ::self.downsample_factor], inpainting_type=self.inpainting_type)
        create_svg(layers, average_colors,
                   filename=output_path, potrace_blur=self.potrace_blur, potrace_opttolerance=self.potrace_opttolerance)
        self.layers = layers
