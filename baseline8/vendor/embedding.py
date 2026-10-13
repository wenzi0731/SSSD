"""Same sinusoidal embedding as SSSD utils/util.py, without hard-coded cuda()."""
import math
import torch


def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    dim = int(diffusion_step_embed_dim_in)
    if dim < 4 or dim % 2:
        raise ValueError("Diffusion embedding dimension must be even and >= 4")
    half = dim // 2
    frequencies = torch.exp(torch.arange(half, device=diffusion_steps.device, dtype=torch.float32)
                            * (-math.log(10000) / (half - 1)))
    angles = diffusion_steps * frequencies
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
