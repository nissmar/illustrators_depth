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

import PIL
import os
import numpy as np
import re
from tqdm import tqdm
from PIL import Image
from svgpathtools import disvg, svg2paths
from io import BytesIO, StringIO


def clean_fill_attributes(svg_content):
    # This function replaces all fill="url(...)" with fill="url(...) #000"
    svg_content = re.sub(r'fill=""', 'fill="#000"', svg_content)
    pattern = r'"(url\([^)]+\))"'
    replacement = r'"#000"'
    return re.sub(pattern, replacement, svg_content)


def make_crisp_edges(svg_file: str):
    # svg_file = re.sub(r'\s*shape-rendering="[^"]*"', '', svg_file)
    if "></path>" in svg_file:
        return svg_file.replace("></path>", ' shape-rendering="crispEdges" ></path>')
    return svg_file.replace("/>", ' shape-rendering="crispEdges" />')


def svg_to_depth_svg(svg_file: str, make_crisp=True, squash_consecutive_same_color=False):
    attr_values = re.findall(r'(?:fill|stroke)="([^"]*)"', svg_file)
    if len(attr_values) >= 255**3:
        print("WARNING: Too many colors")

    def replace_by_index(match):
        attr = match.group(1)
        value = match.group(2)
        if value != "none":
            idx = attr_values.index(value)
            if squash_consecutive_same_color and idx < len(attr_values) - 1:
                if attr_values[idx+1] != attr_values[idx]:
                    # attr_values[idx] = None # previously used this
                    for k in range(idx+1):
                        attr_values[k] = None
            else:
                attr_values[idx] = None
            idx = idx + 1
            return f'{attr}="rgb({idx % 255}, {(idx//255) % 255}, {(idx//255//255) % 255})"'
        else:
            idx = attr_values.index(value)
            attr_values.pop(idx)
            return f'{attr}="{value}"'

    svg_file_indexed = re.sub(
        r'(fill|stroke)="([^"]*)"', replace_by_index, svg_file)

    if make_crisp:
        # remove antialiasing
        svg_file_indexed = make_crisp_edges(svg_file_indexed)
    return svg_file_indexed


def rasterize_svg(svg_string, resolution=256, background_color=None):
    import cairosvg
    svg_raster_bytes = cairosvg.svg2png(
        bytestring=svg_string,
        background_color=background_color,
        output_width=resolution,
        output_height=resolution)
    svg_raster = Image.open(BytesIO(svg_raster_bytes))
    return svg_raster


def rasterize_depth_svg(depth_svg_file, resolution=256):
    raster = rasterize_svg(depth_svg_file, resolution, None)
    depth = np.array(raster).astype(int)
    return depth[..., 0] + depth[..., 1] * 255 + depth[..., 2] * 255 * 255


def transform_svg(svg_str, scale_range=(0.2, 2.), rotation_range=(-180, 180), position_range=(-100, 100), origin=(100, 100)):
    paths, attributes = svg2paths(StringIO(svg_str))
    # Example: Scale all paths by a factor of 2
    new_paths = []
    rotation = np.random.uniform(*rotation_range)
    scale = np.random.uniform(*scale_range)
    position = complex(np.random.uniform(*position_range),
                       np.random.uniform(*position_range))
    for path in paths:
        # print(path)
        path = path.rotated(rotation, complex(*origin))
        path = path.scaled(scale, origin=complex(*origin))
        path = path.translated(position)
        new_paths.append(path)
    return new_paths, attributes
