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

from src.model.illustrators_depth_model import load_illustrators_depth_model
import matplotlib.pyplot as plt
import gradio as gr
from PIL import Image
import torch

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
print('Using device:', DEVICE)


def load_model():
    model, cfg = load_illustrators_depth_model(
        'checkpoints/mmsvg_model/checkpoints/id_model.ckpt', device=DEVICE)
    return model


def depth_prediction(image, apply_cmap=True, blur_sigma=0.0, preserve_size=True):
    image = Image.fromarray(image)

    np_img, pred_depth = model.infer_single_image(
        image,
        preserve_size=True,
        gaussian_sigma=blur_sigma  # ← controlled by slider
    )

    # Normalize depth
    pred_depth = (pred_depth - pred_depth.min()) / (
        pred_depth.max() - pred_depth.min()
    )

    # Apply colormap conditionally
    if apply_cmap:
        return cm(pred_depth)
    else:
        return pred_depth


demo = gr.Interface(
    inputs=[
        gr.Image(height=500),
        gr.Checkbox(label="Apply colormap (inferno)", value=True),
        gr.Slider(
            minimum=0.0,
            maximum=10.0,
            value=0.0,
            step=0.1,
            label="Blur (Gaussian sigma)"
        )
    ],
    outputs=gr.Image(height=500),
    api_name="predict",
    fn=depth_prediction,
    flagging_mode='auto'
)

print('Loading model...')
model = load_model()
cm = plt.cm.get_cmap('inferno')

demo.launch(max_file_size="10mb")
