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

from datetime import datetime
import json
from PIL import Image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
from importlib import import_module
import skimage


def make_prompts_output_dir(pipe_name):
    """Make output directory for prompts"""
    today = datetime.today()
    month = today.month
    day = today.day
    output_dir_name = f"outputs/{pipe_name}_{day}_{month}"
    if not os.path.exists(output_dir_name):
        os.makedirs(output_dir_name)
    return output_dir_name


def natural_sort(list_):
    def natural_key(string_):
        """Sort strings using natural sort order (e.g., file2 before file10)."""
        return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_)]

    return sorted(list_, key=natural_key)


def save_image_and_prompt(image, pipe_args, output_dir_name, size=(512, 512)):
    """Save image and prompt"""
    now = datetime.now()
    current_time = f"{now.hour:02d}:{now.minute:02d}:{now.second:02d}"
    # save pipe args
    pipe_args_json_path = f"{output_dir_name}/{current_time}.json"
    with open(pipe_args_json_path, "w", encoding="utf-8") as f:
        json.dump(pipe_args, f, ensure_ascii=False, indent=2)
    # save image
    prompt = pipe_args["prompt"].split(".")[0]
    image_path = f"{output_dir_name}/{current_time}_{prompt}.png"
    image.resize(size).save(image_path)
    return image.resize(size), image_path


def combine_images(images):
    """Display all images side by side"""
    width, height = images[0].size
    total_width = width * len(images)
    combined_image = Image.new("RGB", (total_width, height))
    for idx, img in enumerate(images):
        combined_image.paste(img, (idx * width, 0))
    return combined_image


# IMAGE PROCESSING


def posterize_with_palette(img, n_colors=8):
    """WARNING: SIMPE VERSION USING KMEANS. Posterize an image using k-means color palette with OpenCV."""
    # Convert to uint8 if needed
    if img.dtype != np.uint8:
        img_uint8 = (
            (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        )
    else:
        img_uint8 = img

    # Reshape image to a 2D array of pixels
    Z = img_uint8.reshape((-1, 3))
    Z = np.float32(Z)

    # Define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = n_colors
    _, label, center = cv2.kmeans(
        Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Convert back to uint8 and make original image shape
    center = np.uint8(center)
    res = center[label.flatten()]
    posterized = res.reshape((img_uint8.shape))
    return posterized, label.reshape(img_uint8.shape[:2]), center


def extract_clusters_remove_thin_components(
    binary_image, kernel_iterations=2, area_threshold=30
):
    mask_np = binary_image.astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_np, connectivity=8
    )
    filtered_mask = np.zeros(binary_image.shape, dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clusters = []
    morph_clusters = []
    for i in range(1, num_labels):  # skip background
        cluster = labels == i
        morph_clust = cv2.morphologyEx(
            cluster.astype(np.uint8) * 255,
            cv2.MORPH_OPEN,
            kernel,
            iterations=kernel_iterations,
        )
        if area_threshold * 255 < morph_clust.sum():
            clusters.append(cluster)
            morph_clusters.append(morph_clust > 125)
            filtered_mask[cluster] = 1
    return filtered_mask, clusters, morph_clusters


def get_pseudo_depth_image(labels, kernel_iterations=2, area_threshold=30):
    depth_image = np.zeros_like(labels)
    id = 1
    for color_id in np.unique(labels):
        _, _, mcs = extract_clusters_remove_thin_components(
            labels == color_id,
            area_threshold=area_threshold,
            kernel_iterations=kernel_iterations,
        )
        for mc in mcs:
            depth_image[mc] = id
            id += 1
    return depth_image


def add_white_background(img):
    """
    Adds a white background to a PIL image (e.g., to remove transparency).
    Returns a new PIL.Image object.
    """
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        background = Image.new("RGB", img.size, (255, 255, 255))
        # Use alpha channel as mask
        background.paste(img, mask=img.split()[-1])
        return background
    else:
        return img.convert("RGB")



def show_side_by_side(imgs, titles, base_size=4, save_name=None, axs=None, **kwargs):
    assert len(imgs) == len(titles)
    if axs is None:
        _, axs = plt.subplots(1, len(imgs), figsize=(
            base_size * len(imgs), base_size))
    for i in range(len(imgs)):
        axs[i].imshow(imgs[i], **kwargs)
        axs[i].set_title(titles[i])
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name)
    plt.show()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normalize_depth(depth):
    return (depth - depth.min()) / (depth.max() - depth.min())


def depth_to_point_cloud(filename, rgb_image, depth_image):
    import open3d as o3d

    h, w = depth_image.shape
    i_indices, j_indices = np.meshgrid(
        np.arange(h), np.arange(w), indexing="ij")
    reg_depth = h * depth_image.flatten()
    points = np.stack(
        [i_indices.flatten(), j_indices.flatten(), reg_depth], axis=-1
    ).reshape(-1, 3)
    colors = rgb_image.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True)


def resolve_path(path):
    module_path, name = path.rsplit(".", 1)
    return getattr(import_module(module_path), name)


def clean_segments(labels, kernel_iterations=2, area_threshold=10):
    depth_image = np.zeros_like(labels)
    id = 1
    for color_id in np.unique(labels):
        _, mcs, _ = extract_clusters_remove_thin_components(
            labels == color_id,
            area_threshold=area_threshold,
            kernel_iterations=kernel_iterations,
        )
        for mc in mcs:
            depth_image[mc] = id
            id += 1
    return depth_image


def segment_and_clean_depth(depth, cut_threshold=.1, kernel_iterations=2, area_threshold=10, normalize=True):
    if normalize:
        depth = (depth - depth.min()) / (depth.max() - depth.min())
    img = depth[..., None].repeat(3, -1)
    segments_quick = skimage.segmentation.quickshift(
        img, kernel_size=3, max_dist=6, ratio=0.5, channel_axis=-1)

    g = skimage.graph.rag_mean_color(
        img,
        segments_quick
    )
    seg = skimage.graph.cut_threshold(segments_quick,
                                      g,
                                      thresh=cut_threshold)
    seg = clean_segments(
        seg, kernel_iterations=kernel_iterations, area_threshold=area_threshold)
    new_depth = depth.copy()
    for i in np.unique(seg):
        new_depth[seg == i] = np.median(depth[seg == i])
    new_depth[seg == 0] = 0
    return new_depth


def interactive_layer_slider(layers, average_colors):
    import ipywidgets

    stacked_imgs = []
    final_img = np.zeros((*layers[0].shape, 3))
    for layer, color in zip(layers, (average_colors)):
        final_img[layer] = color
        stacked_imgs.append(final_img.copy())

    def f(x):
        plt.imshow(stacked_imgs[x])
        plt.show()
    return ipywidgets.interact(f, x=ipywidgets.widgets.IntSlider(min=0, max=len(stacked_imgs)-1, step=1, value=1))
