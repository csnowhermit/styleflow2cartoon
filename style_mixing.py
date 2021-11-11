# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate style mixing image matrix using pretrained network pickle."""

import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy
import functools
import random


# ----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    vals = s.split(',')
    return [int(x) for x in vals]


# ----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--rows', 'row_seeds', type=num_range, help='Random seeds to use for image rows', required=True)
@click.option('--cols', 'col_seeds', type=num_range, help='Random seeds to use for image columns', required=True)
@click.option('--styles', 'col_styles', type=num_range, help='Style layer range', default='0-6', show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const',
              show_default=True)
@click.option('--outdir', type=str, required=True)
@click.option('--gpu', help='Use GPU to generate image (using CPU if not specified)', default='False', type=str,
              show_default=True)
@click.option('--seed-size', 'seed_size', help='The range at which the seed can be chosen from', default=500000,
              type=int, show_default=True)
def generate_style_mix(
        network_pkl: str,
        row_seeds: List[int],
        col_seeds: List[int],
        col_styles: List[int],
        truncation_psi: float,
        noise_mode: str,
        outdir: str,
        # gpu: List[str],
        gpu: Optional[str],  # default: False
        seed_size: List[str]
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    python style_mixing.py --outdir=out --rows=85,100,75,458,1500 --cols=55,821,1789,293 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """
    print('Loading networks from "%s"...' % network_pkl)
    if gpu == "True":
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    print("G:", type(G))
    if gpu != "True":
        # G.forward = functools.partial(G.forward, force_fp32=True)    # cpu下强制用32位浮点数
        G = G.float()

    os.makedirs(outdir, exist_ok=True)

    print('Generating W vectors...')
    all_seeds = list(set(row_seeds + col_seeds))
    all_z = np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])  # [len(all_seeds), 512]
    all_w = G.mapping(torch.from_numpy(all_z).to(device), None)  # torch.Tensor, [len(all_seeds), 18, 512]
    w_avg = G.mapping.w_avg  # [512]
    all_w = w_avg + (all_w - w_avg) * truncation_psi  # [len(all_seeds), 18, 512]
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))}  # 每个seed对应的all_w为[18, 512]

    import cv2
    print('Generating images...')
    all_images = G.synthesis(all_w, noise_mode=noise_mode,
                             force_fp32=True)  # noise_mode为const all_images[len(all_seeds), 3, 1024, 1024]
    all_images = (all_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
    image_dict = {(seed, seed): image for seed, image in zip(all_seeds, list(all_images))}
    # for img in list(all_images):
    #     cv2.imshow("img", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    #     cv2.waitKey()

    print('Generating style-mixed images...')
    for row_seed in row_seeds:
        for col_seed in col_seeds:
            w = w_dict[row_seed].clone()  # [18, 512]
            w[col_styles] = w_dict[col_seed][col_styles]  # 拼接新的style w[col_styles] [7, 512]
            print(w[np.newaxis].shape)
            image = G.synthesis(w[np.newaxis], noise_mode=noise_mode, force_fp32=True)  # w[np.newaxis], [1, 18, 512]
            print(type(image), type(w[np.newaxis]))
            image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            print(type(image))
            image_dict[(row_seed, col_seed)] = image[0].cpu().numpy()

            # cv2.imshow("%s_%s" % (str(row_seed), str(col_seed)), cv2.cvtColor(image[0].cpu().numpy(), cv2.COLOR_RGB2BGR))
            # cv2.waitKey()

    print('Saving images...')
    os.makedirs(outdir, exist_ok=True)
    for (row_seed, col_seed), image in image_dict.items():
        PIL.Image.fromarray(image, 'RGB').save(f'{outdir}/{row_seed}-{col_seed}.png')

    print('Saving image grid...')
    W = G.img_resolution
    H = G.img_resolution
    canvas = PIL.Image.new('RGB', (W * (len(col_seeds) + 1), H * (len(row_seeds) + 1)), 'black')
    for row_idx, row_seed in enumerate([0] + row_seeds):
        for col_idx, col_seed in enumerate([0] + col_seeds):
            if row_idx == 0 and col_idx == 0:
                continue
            key = (row_seed, col_seed)
            if row_idx == 0:
                key = (col_seed, col_seed)
            if col_idx == 0:
                key = (row_seed, row_seed)
            canvas.paste(PIL.Image.fromarray(image_dict[key], 'RGB'), (W * col_idx, H * row_idx))
    canvas.save(f'{outdir}/grid.png')


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_style_mix()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
