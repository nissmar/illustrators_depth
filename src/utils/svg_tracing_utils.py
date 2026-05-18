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
import potrace
import imageio
from scipy.ndimage import distance_transform_edt
import skimage
from skimage.restoration import inpaint_biharmonic
from skimage.morphology import convex_hull_image, binary_erosion, disk, binary_dilation
from tqdm import tqdm
import vtracer
import os


def quantize_and_sort_colors(img, quantization_bits=4):
    print("WARNING: color quantization may fail.")
    q_level = 2**quantization_bits
    q_img = (img * q_level).round().astype(int)
    q_img = q_img[..., 0] + q_level * q_img[..., 1] + \
        q_level * q_level * q_img[..., 2]
    unique_vals, counts = np.unique(q_img.flatten(), return_counts=True)
    sorted_indices = np.argsort(counts)
    unique_sorted = unique_vals[sorted_indices]
    # counts_sorted = counts[sorted_indices]
    return q_img, unique_sorted


def clean_thin_structures(mask, erosion_steps, dilation_steps):
    assert mask.dtype == bool
    eroded_mask = binary_erosion(mask, footprint=disk(erosion_steps))
    dilated_mask = binary_dilation(eroded_mask, footprint=disk(dilation_steps))
    new_mask = mask.copy()
    new_mask[~dilated_mask] = 0
    return new_mask


def clean_segments(segments, erosion_steps=3, dilation_steps=6):
    new_segments = np.zeros_like(segments)
    for idx in np.unique(segments):
        clean_segment = clean_thin_structures(
            segments == idx, erosion_steps, dilation_steps)
        new_segments[clean_segment] = idx+1
    return new_segments


def cluster_and_smooth_felzenszwalb(img, scale=100, sigma=0.01, min_size=50, erosion_steps=1, dilation_steps=2):
    segments_fz = skimage.segmentation.felzenszwalb(
        img, scale=scale, sigma=sigma, min_size=min_size)
    clean_segments_fz = clean_segments(
        segments_fz, erosion_steps, dilation_steps)
    clean_segments_fz = fill_palette_idx_zeros_with_closest(clean_segments_fz)
    return clean_segments_fz


def cluster_and_smooth_quickshift(img, kernel_size=5, max_dist=10, ratio=1., sigma=0, cut_threshold=0.1, erosion_steps=1, dilation_steps=2):
    segments_quick = skimage.segmentation.quickshift(
        img, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio, sigma=sigma)
    g = skimage.graph.rag_mean_color(
        img,
        segments_quick
    )
    rgb_seg = skimage.graph.cut_threshold(segments_quick,
                                          g,
                                          thresh=cut_threshold)
    clean_segments_quick = clean_segments(
        rgb_seg, erosion_steps, dilation_steps)
    clean_segments_quick = fill_palette_idx_zeros_with_closest(
        clean_segments_quick)
    return clean_segments_quick


def map_colors(
    img, quantization_bits=4, erosion_threshold=0.1, erode_iterations=1, max_colors=50
):
    """quantize image, find thick regions (that resist to erosion), and return palette_idx"""
    q_img, unique_sorted = quantize_and_sort_colors(img, quantization_bits)
    palette_idx = np.zeros_like(q_img)
    for i in range(1, min(max_colors, len(unique_sorted))):
        mask = q_img == unique_sorted[-i]
        eroded_mask = binary_erosion(mask, footprint=disk(erode_iterations+1))
        if (eroded_mask.sum() / mask.sum()) > erosion_threshold:
            palette_idx[mask] = i
    return palette_idx


def fill_palette_idx_zeros_with_closest(palette_idx):
    """
    Fill zeros in palette_idx with the value of the closest nonzero pixel.
    """
    mask = palette_idx == 0
    if not np.any(mask):
        return palette_idx.copy()
    # Find indices of nearest nonzero for each zero pixel
    # distance_transform_edt returns, for each zero pixel, the indices of the closest nonzero pixel
    filled = palette_idx.copy()
    # For each zero pixel, get the nearest nonzero pixel's value
    nearest = distance_transform_edt(
        mask, return_distances=False, return_indices=True)
    filled[mask] = palette_idx[tuple(nearest[:, mask])]
    return filled


def average_colors_on_regions(palette_clean_idx, img):
    """
    Average colors on the regions of the palette_clean_idx
    """
    average_colors = np.zeros_like(img)
    for i in np.unique(palette_clean_idx):
        average_colors[palette_clean_idx == i] = np.median(
            img[palette_clean_idx == i],
            axis=0
        )
    return average_colors


def average_depth_on_regions(palette_clean_idx, predicted_depth):
    """
    Average the depth on the regions of the palette_clean_idx
    """
    average_depth = np.zeros_like(predicted_depth)
    for i in np.unique(palette_clean_idx):
        average_depth[palette_clean_idx == i] = np.median(
            predicted_depth[palette_clean_idx == i]
        )
    return average_depth


def inpaint_convex(current_mask, removed):
    convex_n_p = convex_hull_image(current_mask)
    current_mask[removed] = convex_n_p[removed]
    convexity_score = (current_mask[convex_n_p == True]).mean()
    return current_mask, convexity_score


def inpaint_harmonic(current_mask, removed):
    current_mask[removed] = 0
    current_mask = inpaint_biharmonic(1.0*current_mask, removed) > .5
    return current_mask


def extract_and_inpaint_layers(palette_clean, inpainting_type='mix', cvx_threshold=0.8):
    """extract the layers from the clean palette_clean, and inpaint them with the closest pixel color"""
    idx = np.unique(palette_clean)
    removed = np.zeros_like(palette_clean).astype(bool)
    layers = []
    for i in tqdm(idx[::-1]):
        if inpainting_type == 'closest':
            n_p = palette_clean.copy()
            n_p[removed] = 0
            n_p = fill_palette_idx_zeros_with_closest(n_p)
            layers.append(n_p == i)
        elif inpainting_type == 'harmonic':
            n_p = palette_clean.copy() == i
            layers.append(inpaint_harmonic(n_p, removed))
        elif inpainting_type == 'convex':
            n_p = palette_clean.copy() == i
            n_p, convexity_score = inpaint_convex(n_p, removed)
            layers.append(n_p)
        elif inpainting_type == 'mix':
            n_p = palette_clean.copy() == i
            cvx_try, convexity_score = inpaint_convex(n_p, removed)
            if convexity_score > cvx_threshold:
                layers.append(cvx_try)
            else:
                layers.append(inpaint_harmonic(n_p, removed))
        else:
            raise ValueError(f"Invalid type: {inpainting_type}")
        removed[palette_clean == i] = True
    layers.reverse()
    return layers


def simplify_depth(average_depth, img, color_dist_threshold=0.1):
    """
    Change the depth values to the number of the region, merge regions with similar colors. TODO better sorting strategy and better color selection.
    """
    print("WARNING: very simple merging strategy and color averaging.")
    simplified_depth = np.zeros_like(average_depth).astype(int)
    prev_color = None
    depth_idx = 0
    average_colors = []
    for a_d in np.unique(average_depth):
        average_color = np.median(img[average_depth == a_d], axis=0)
        depth_idx += 1
        average_colors.append(average_color)
        if prev_color is not None:
            dist = np.linalg.norm(average_color - prev_color)
            # print(dist)
            if dist < color_dist_threshold:
                depth_idx -= 1
                average_colors.pop()
        simplified_depth[average_depth == a_d] = depth_idx
        prev_color = average_color
    return simplified_depth, average_colors


# IO


def create_svg(layers, average_colors, filename="output", potrace_blur=0, potrace_opttolerance=0.2, **kwargs):
    """create an svg file from the layers using potrace. kwargs: bm.trace()"""
    w, h = layers[0].shape
    filename = filename.replace(".svg", "")
    with open(f"{filename}.svg", "w") as fp:
        fp.write(
            f"""<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{w}" height="{h}" viewBox="0 0 {w} {h}">"""
        )
        for layer, color in zip(layers, average_colors):
            this_layer = np.logical_not(layer)
            if potrace_blur > 0:
                this_layer = skimage.filters.gaussian(
                    this_layer.astype(float), sigma=potrace_blur) > .5
            bm = potrace.Bitmap(this_layer)
            plist = bm.trace(**kwargs,
                             opttolerance=potrace_opttolerance
                             )
            #     turdsize=2,
            #     turnpolicy=POTRACE_TURNPOLICY_MINORITY,
            #     alphamax=1,
            #     opticurve=False,
            #     opttolerance=0.2,
            parts = []
            for curve in plist:
                fs = curve.start_point
                parts.append(f"M{fs.x},{fs.y}")
                for segment in curve.segments:
                    if segment.is_corner:
                        a = segment.c
                        b = segment.end_point
                        parts.append(f"L{a.x},{a.y}L{b.x},{b.y}")
                    else:
                        a = segment.c1
                        b = segment.c2
                        c = segment.end_point
                        parts.append(f"C{a.x},{a.y} {b.x},{b.y} {c.x},{c.y}")
                parts.append("z")
            fp.write(
                f'<path stroke="none" fill="rgb({color[0]*255},{color[1]*255},{color[2]*255})" fill-rule="evenodd" d="{"".join(parts)}"/>'
            )
        fp.write("</svg>")


def create_gif(layers, average_colors, filename="output", duration=500):
    final_img = np.zeros((*layers[0].shape, 3))
    gif_images = []
    for layer, color in zip(layers, average_colors):
        final_img[layer] = color
        gif_images.append(final_img.copy())
    # Convert images to uint8 and append copies to avoid in-place modification
    gif_images_uint8 = [(255 * img).copy().astype(np.uint8)
                        for img in gif_images]
    # Save as animated GIF
    imageio.mimsave(f"{filename}.gif", gif_images_uint8,
                    duration=duration, loop=0)


def create_html(src_folder, i_max, i_min=0):
    # Create HTML
    html = f"""
    <html>
    <head>
    <style>
        .img-row {{
        display: flex;
        flex-direction: row;
        gap: 30px;
        align-items: flex-start;
        }}
        .img-col {{
        display: flex;
        flex-direction: column;
        align-items: center;
        }}
        .img-col img {{
        border: 1px solid #888;
        max-width: 256px;
        margin-bottom: 8px;
        }}
        .img-col label {{
        font-size: 1.1em;
        margin-bottom: 4px;
        }}
    </style>
    </head>
    <body>
    {''.join(f'''
    <div class="img-row">
        <div class="img-col">
        <label>Input Image {i}</label>
        <img src="{i}_input.png" />
        </div>
        <div class="img-col">
        <label>Predicted Depth</label>
        <img src="{i}_predicted_depth.png" />
        </div>
        <div class="img-col">
        <label>Animation</label>
        <img src="{i}_output.gif" />
        </div>
        <div class="img-col">
        <label>Input Image {i+1}</label>
        <img src="{i+1}_input.png" />
        </div>
        <div class="img-col">
        <label>Predicted Depth</label>
        <img src="{i+1}_predicted_depth.png" />
        </div>
        <div class="img-col">
        <label>Animation</label>
        <img src="{i+1}_output.gif" />
        </div>
    </div>
    ''' for i in range(i_min, i_max, 2))}
    </body>
    </html>
    """
    with open(f"{src_folder}/00_index.html", "w") as f:
        f.write(html)
    return html


def vtracer_trace(input_path, output_path='tmp/test_temp_vtracer.svg', save=False, **kwargs):
    vtracer.convert_image_to_svg_py(input_path,
                                    output_path,
                                    **kwargs
                                    )
    with open(output_path, "r") as f:
        svg_str = f.read()
    if not save:
        os.remove(output_path)
    return svg_str
