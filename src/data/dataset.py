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
from torch.utils.data import Dataset
import numpy as np
import os
from tqdm import tqdm
import lightning as pL
from torchvision import transforms
from src.utils.prompt_utils import natural_sort, resolve_path
from src.data.data_utils import clean_fill_attributes, svg_to_depth_svg, make_crisp_edges, rasterize_svg, rasterize_depth_svg, transform_svg
from PIL import Image
from svgpathtools import disvg


def tensor_to_numpy_rgb(im, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    im = im.cpu().detach().numpy().squeeze().transpose(1, 2, 0)
    im = (im) * np.array(std) + np.array(mean)
    return im.clip(0, 1)


def custom_transforms(input_size=1536, data_aug=None, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    trs = [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    if data_aug is not None:
        tforms = []
        for t in data_aug:
            trans = resolve_path(t)
            kwargs = data_aug[t]
            tforms.append(trans(**kwargs))
        trs = tforms + trs
    return transforms.Compose(trs)


def process_pil(rgb_image, return_size=False):
    original_size = rgb_image.size
    transforms = custom_transforms()
    tensor_img = transforms(rgb_image).unsqueeze(0)
    np_img = tensor_to_numpy_rgb(
        tensor_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    if return_size:
        return tensor_img, np_img, original_size
    return tensor_img, np_img


def load_single_image(path, return_size=False):
    rgb_image = Image.open(path).convert('RGB')
    original_size = rgb_image.size
    transforms = custom_transforms()
    tensor_img = transforms(rgb_image).unsqueeze(0)
    np_img = tensor_to_numpy_rgb(
        tensor_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    if return_size:
        return tensor_img, np_img, original_size
    return tensor_img, np_img


class RGBDepthDataset(Dataset):
    def __init__(
        self,
        data,
        img_transforms,
        raster_resolution=256,
        min_files=0,
        max_files=100000,
        idx_files=None,
        squash_consecutive_same_color=False,
        make_crisp_input=False,
        background_color="white",
        concat_n=None,
    ):
        self.img_transforms = img_transforms
        if idx_files is not None:
            indices = np.load(idx_files)
            indices = indices.astype(int).tolist()
        else:
            indices = range(min_files, max_files)
        self.svg_files = [clean_fill_attributes(
            data[i]['svg']) for i in tqdm(indices)]
        self.squash_consecutive_same_color = squash_consecutive_same_color
        self.make_crisp_input = make_crisp_input
        self.raster_resolution = raster_resolution
        self.background_color = background_color
        if concat_n is not None:
            print(f"Concatenating {concat_n} SVGs")
            self.concat_n = concat_n
        else:
            self.concat_n = False
        print(f"Found {len(self.svg_files)} files")

    def __len__(self):
        return len(self.svg_files)

    def get_concat_svg(self, start_idx):
        full_paths = []
        full_attributes = []
        for i in range(start_idx, start_idx+self.concat_n):
            svg_str = self.svg_files[i % self.__len__()]
            # im = np.array(rasterize_svg(make_crisp_edges(svg_str), resolution=50))
            # if len(np.unique(im)) > 2:
            new_paths, attributes = transform_svg(svg_str)
            full_paths.extend(new_paths)
            full_attributes.extend(attributes)
            # if len(full_paths) > 0:
            filename = f'/mnt/localssd/{start_idx}.svg'
            disvg(full_paths, attributes=full_attributes,
                  openinbrowser=False, filename=filename, dimensions=(200, 200))
            with open(filename, 'r') as f:
                new_svg = f.read()
            os.system(f'rm {filename}')
        return new_svg

    def __getitem__(self, idx):
        if self.concat_n:
            svg_file = self.get_concat_svg(idx)
        else:
            svg_file = self.svg_files[idx]
        depth_svg_file = svg_to_depth_svg(
            svg_file, make_crisp=True, squash_consecutive_same_color=self.squash_consecutive_same_color)  # crisp by default
        if self.make_crisp_input:
            svg_file = make_crisp_edges(svg_file)

        # rgb_image
        rgb_img = rasterize_svg(
            svg_file, resolution=self.raster_resolution, background_color=self.background_color)
        img_norm = self.img_transforms(rgb_img)

        # depth_image
        depth_img = rasterize_depth_svg(
            depth_svg_file, resolution=self.raster_resolution)
        _, ngt = np.unique(depth_img, return_inverse=True)
        depth_img = ngt.reshape(depth_img.shape)
        depth_img = torch.from_numpy(depth_img)

        return img_norm, depth_img


class DepthData(pL.LightningDataModule):
    def __init__(
        self,
        data,
        input_size=1024,
        raster_resolution=256,
        squash_consecutive_same_color=False,
        make_crisp_input=False,
        batch_size=10,
        num_workers=10,
        shuffle=True,
        min_files=0,
        max_files=None,
        idx_files=None,
        background_color="white",
        transforms=None,
        concat_n=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.transforms = transforms
        self.dataset = RGBDepthDataset(
            data,
            self.transforms,
            raster_resolution=raster_resolution,
            min_files=min_files,
            max_files=max_files,
            idx_files=idx_files,
            background_color=background_color,
            squash_consecutive_same_color=squash_consecutive_same_color,
            make_crisp_input=make_crisp_input,
            concat_n=concat_n,
        )

    def dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )
