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

src_folder = "./data/svg_examples/svg_ref/"
files = os.listdir(src_folder)
for file in files:
    if file.endswith(".svg"):
        if not ("resized" in file):
            print(file)
            os.system(
                f"cairosvg -f png --output-width 1024 --output-height 1024 -b white -i {src_folder}/{file} -o {src_folder}/{file.replace('.svg', '.png')}"
            )
            os.system(
                f"cairosvg -f svg --output-width 1024 --output-height 1024 -i {src_folder}/{file} -o {src_folder}/{file.replace('.svg', '_resized.svg')}"
            )
