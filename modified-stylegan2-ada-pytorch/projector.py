# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import os
from time import perf_counter

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

import dnnlib
import legacy

def project(
    G,
    target: torch.Tensor,  # [C,H,W] and dynamic range [0,255]
    *,
    num_steps: int = 1000,
    w_avg_samples: int = 10000,
    initial_learning_rate: float = 0.1,
    initial_noise_factor: float = 0.05,
    lr_rampdown_length: float = 0.25,
    lr_rampup_length: float = 0.05,
    noise_ramp_length: float = 0.75,
    regularize_noise_weight: float = 1e5,
    verbose: bool = False,
    device: torch.device
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)

    # Compute w stats on GPU
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples on {device}...')
    z = torch.randn([w_avg_samples, G.z_dim], device=device)
    w = G.mapping(z, None)                # [N, L, C]
    w = w[:, :1, :]                       # [N, 1, C]
    w_avg = w.mean(dim=0, keepdim=True)
    w_std = ((w - w_avg).pow(2).sum() / w_avg_samples).sqrt()

    # Setup noise inputs
    noise_bufs = {
        name: buf
        for (name, buf) in G.synthesis.named_buffers()
        if 'noise_const' in name
    }

    # Load VGG16 feature detector
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Prepare target features
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256,256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    # Optimize in W space
    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True)
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()),
                                 betas=(0.9, 0.999),
                                 lr=initial_learning_rate)

    # Initialize noise buffers
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Generate synth images
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, noise_mode='const')

        # Downsample for VGG16 if needed
        synth_images = (synth_images + 1) * (255/2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256,256), mode='area')

        # Compute distance
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:]
            while True:
                reg_loss += (noise * torch.roll(noise, 1, 3)).mean()**2
                reg_loss += (noise * torch.roll(noise, 1, 2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, 2)
        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step+1}/{num_steps}: dist {dist:.2f} loss {float(loss):.2f}')

        w_out[step] = w_opt.detach()[0]

        # Re-normalize noise
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out.repeat([1, G.mapping.num_ws, 1])

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--target', 'target_fname', help='Target image to project', required=True, metavar='FILE')
@click.option('--num-steps',   type=int, default=1000, show_default=True)
@click.option('--seed',        type=int, default=303,  show_default=True)
@click.option('--save-video',  type=bool, default=True, show_default=True)
@click.option('--outdir',      help='Directory for outputs', required=True, metavar='DIR')
def run_projection(network_pkl: str, target_fname: str, outdir: str,
                   save_video: bool, seed: int, num_steps: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f'Loading network from "{network_pkl}"...')
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].to(device)

    # Prepare target image
    img = PIL.Image.open(target_fname).convert('RGB')
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2))
    img = img.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    arr = np.array(img, dtype=np.uint8)

    # Run projection
    start = perf_counter()
    w_steps = project(G, target=torch.tensor(arr.transpose([2,0,1]), device=device),
                      num_steps=num_steps, device=device, verbose=True)
    print(f'Elapsed: {perf_counter()-start:.1f}s')

    os.makedirs(outdir, exist_ok=True)
    if save_video:
        video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=10,
                                   codec='libx264', bitrate='16M')
        for w_i in w_steps:
            im = (G.synthesis(w_i.unsqueeze(0), noise_mode='const') + 1)*(255/2)
            im = im.permute(0,2,3,1).clamp(0,255).to(torch.uint8)[0].cpu().numpy()
            video.append_data(np.concatenate([arr, im], axis=1))
        video.close()

    # Save final results
    PIL.Image.fromarray(im, 'RGB').save(f'{outdir}/proj.png')
    np.savez(f'{outdir}/projected_w.npz', w=w_steps[-1].unsqueeze(0).cpu().numpy())

if __name__ == "__main__":
    run_projection()