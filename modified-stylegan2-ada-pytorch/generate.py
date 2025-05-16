# # Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# #
# # NVIDIA CORPORATION and its licensors retain all intellectual property
# # and proprietary rights in and to this software, related documentation
# # and any modifications thereto.  Any use, reproduction, disclosure or
# # distribution of this software and related documentation without an express
# # license agreement from NVIDIA CORPORATION is strictly prohibited.

# """Generate images using pretrained network pickle."""

# import os
# import re
# from typing import List, Optional

# import click
# import dnnlib
# import numpy as np
# import PIL.Image
# import torch

# import legacy
# import time
# import csv

# #----------------------------------------------------------------------------

# def num_range(s: str) -> List[int]:
#     '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

#     range_re = re.compile(r'^(\d+)-(\d+)$')
#     m = range_re.match(s)
#     if m:
#         return list(range(int(m.group(1)), int(m.group(2))+1))
#     vals = s.split(',')
#     return [int(x) for x in vals]

# #----------------------------------------------------------------------------

# @click.command()
# @click.pass_context
# @click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
# @click.option('--seeds', type=num_range, help='List of random seeds')
# @click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
# @click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
# @click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
# @click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
# @click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
# def generate_images(
#     ctx: click.Context,
#     network_pkl: str,
#     seeds: Optional[List[int]],
#     truncation_psi: float,
#     noise_mode: str,
#     outdir: str,
#     class_idx: Optional[int],
#     projected_w: Optional[str]
# ):
#     """Generate images using pretrained network pickle.

#     Examples:

#     \b
#     # Generate curated MetFaces images without truncation (Fig.10 left)
#     python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
#         --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

#     \b
#     # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
#     python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
#         --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

#     \b
#     # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
#     python generate.py --outdir=out --seeds=0-35 --class=1 \\
#         --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

#     \b
#     # Render an image from projected W
#     python generate.py --outdir=out --projected_w=projected_w.npz \\
#         --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
#     """
#     start_time = time.time()
    
#     print('Loading networks from "%s"...' % network_pkl)
#     device = torch.device('cuda')
#     with dnnlib.util.open_url(network_pkl) as f:
#         G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

#     os.makedirs(outdir, exist_ok=True)

#     # Synthesize the result of a W projection.
#     if projected_w is not None:
#         if seeds is not None:
#             print ('warn: --seeds is ignored when using --projected-w')
#         print(f'Generating images from projected W "{projected_w}"')
#         ws = np.load(projected_w)['w']
#         ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
#         assert ws.shape[1:] == (G.num_ws, G.w_dim)
#         for idx, w in enumerate(ws):
#             img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
#             img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
#             img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/proj{idx:02d}.png')
#         return

#     if seeds is None:
#         ctx.fail('--seeds option is required when not using --projected-w')

#     # Labels.
#     label = torch.zeros([1, G.c_dim], device=device)
#     if G.c_dim != 0:
#         if class_idx is None:
#             ctx.fail('Must specify class label with --class when using a conditional network')
#         label[:, class_idx] = 1
#     else:
#         if class_idx is not None:
#             print ('warn: --class=lbl ignored when running on an unconditional network')

#     # Generate images.
#     for seed_idx, seed in enumerate(seeds):
#         print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
#         z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
#         img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
#         img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
#         PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')
        
    
#     end_time = time.time()
#     total_time_sec = end_time - start_time
#     total_time_hours = total_time_sec / 3600

#     # Calculate number of images
#     num_images = len(seeds)

#     # Calculate average inference time
#     average_inference_time_sec = total_time_sec / num_images

#     # Print Performance Summary
#     print(f"\n[Performance Summary]")
#     print(f"Network snapshot       : {os.path.basename(network_pkl)}")
#     print(f"Total images generated : {num_images}")
#     print(f"Total runtime (seconds): {total_time_sec:.2f}")
#     print(f"Average time per image : {average_inference_time_sec:.4f} sec")
#     print(f"Total GPU hours        : {total_time_hours:.4f}")
    
    
#     # Save performance metrics to CSV
#     log_file = os.path.join(outdir, "timing_log.csv")
#     file_exists = os.path.isfile(log_file)

#     with open(log_file, mode='a', newline='') as f:
#         writer = csv.writer(f)
#         if not file_exists:
#             # Header for the CSV
#             writer.writerow([
#                 'network',
#                 'class_idx',
#                 'seeds',
#                 'num_images',
#                 'total_seconds',
#                 'average_time_per_image_sec',
#                 'gpu_hours'
#             ])
            
#             writer.writerow([
#                 os.path.basename(network_pkl),
#                 class_idx if class_idx is not None else "unconditional",
#                 seeds,
#                 num_images,
#                 f"{total_time_sec:.2f}",
#                 f"{average_inference_time_sec:.4f}",
#                 f"{total_time_hours:.4f}"
#             ])

# #----------------------------------------------------------------------------

# if __name__ == "__main__":
#     generate_images() # pylint: disable=no-value-for-parameter

# #----------------------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.

"""Generate images using pretrained network pickle with performance monitoring."""

import os
import re
import time
import csv
import subprocess
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''
    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

def get_gpu_name() -> str:
    '''Get the name of the active GPU.'''
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return "Unknown GPU"

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    class_idx: Optional[int],
    projected_w: Optional[str]
):
    """Generate images using pretrained network pickle and monitor performance."""

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # GPU Info
    gpu_name = get_gpu_name()
    print(f"Using GPU: {gpu_name}")

    # Synthesize the result of a W projection.
    if projected_w is not None:
        if seeds is not None:
            print('warn: --seeds is ignored when using --projected-w')
        print(f'Generating images from projected W "{projected_w}"')
        ws = np.load(projected_w)['w']
        ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        for idx, w in enumerate(ws):
            img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/proj{idx:02d}.png')
        return

    if seeds is None:
        ctx.fail('--seeds option is required when not using --projected-w')

    # Labels
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    # Pre-Run Estimate
    avg_time_per_image_sec = 0.5  # Adjust based on real GPU performance
    num_images = len(seeds)
    estimated_total_seconds = num_images * avg_time_per_image_sec
    estimated_gpu_hours = estimated_total_seconds / 3600

    print(f"\n[Estimate] About {estimated_total_seconds:.2f} seconds (~{estimated_gpu_hours:.4f} GPU hours) needed for {num_images} images.\n")

    # Start Timing
    start_time = time.time()

    # Generate images
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx + 1, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')

    # End Timing
    end_time = time.time()
    total_time_sec = end_time - start_time
    total_time_hours = total_time_sec / 3600
    average_inference_time_sec = total_time_sec / num_images

    # Print Final Performance Summary
    print(f"\n[Performance Summary]")
    print(f"Network snapshot       : {os.path.basename(network_pkl)}")
    print(f"Class label            : {class_idx if class_idx is not None else 'unconditional'}")
    print(f"Total images generated : {num_images}")
    print(f"Total runtime (seconds): {total_time_sec:.2f}")
    print(f"Average time per image : {average_inference_time_sec:.4f} sec")
    print(f"Total GPU hours        : {total_time_hours:.4f}")
    print(f"GPU                    : {gpu_name}")

    # Save performance to CSV
    log_file = os.path.join(outdir, "timing_log.csv")
    file_exists = os.path.isfile(log_file)

    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'network',
                'class_idx',
                'seeds',
                'num_images',
                'total_seconds',
                'average_time_per_image_sec',
                'gpu_hours',
                'gpu_name'
            ])
        writer.writerow([
            os.path.basename(network_pkl),
            class_idx if class_idx is not None else "unconditional",
            seeds,
            num_images,
            f"{total_time_sec:.2f}",
            f"{average_inference_time_sec:.4f}",
            f"{total_time_hours:.4f}",
            gpu_name
        ])

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------