"""HEEW adapter around the original SSSD-S4 denoising architecture.

Only four energy channels diffuse; weather/calendar/year are clean known lanes.
The imputer's projections, two S4 layers per residual block, gating and skip
connections are retained. Loss/scoring never include known condition channels.
"""
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F

from baseline8.vendor.imputer import SSSDS4Imputer
from baseline8.vendor.s4 import HippoSSKernel


class ConditionalSSSD(nn.Module):
    def __init__(self, config, condition_channels):
        super().__init__()
        cfg = dict(config["model"])
        self.condition_channels = int(condition_channels)
        channels = 4 + self.condition_channels + 1
        if int(cfg["s4_lmax"]) != 24 or int(cfg["s4_d_state"]) < 2 or int(cfg["s4_d_state"]) % 2:
            raise ValueError("Require s4_lmax=24 and positive even s4_d_state >= 2")
        if min(int(cfg[k]) for k in ("res_channels", "skip_channels", "num_res_layers")) < 1:
            raise ValueError("Residual/skip/layer counts must be positive")
        self.denoiser = SSSDS4Imputer(in_channels=channels, out_channels=channels, **cfg)
        d = config["diffusion"]
        self.num_steps = int(d["num_steps"])
        if self.num_steps < 2 or not 0 < float(d["beta_start"]) <= float(d["beta_end"]) < 1:
            raise ValueError("Require T>=2 and 0 < beta_start <= beta_end < 1")
        if d["schedule"] != "linear":
            raise ValueError("baseline8 uses upstream linear beta schedule")
        beta = torch.linspace(float(d["beta_start"]), float(d["beta_end"]), self.num_steps)
        alpha = 1 - beta
        abar = torch.cumprod(alpha, dim=0)
        variance = beta.clone()
        variance[1:] *= (1 - abar[:-1]) / (1 - abar[1:])
        for name, value in (("beta", beta), ("alpha", alpha), ("alpha_bar", abar), ("sigma", variance.sqrt())):
            self.register_buffer(name, value)

    def pack_conditions(self, conditions, year):
        if conditions.ndim != 3 or conditions.shape[1:] != (self.condition_channels, 24):
            raise ValueError("Expected conditions [B,K,24]")
        if year.shape != (len(conditions), 1, 24):
            raise ValueError("Expected year [B,1,24]")
        known = torch.cat([conditions, year], dim=1)
        zeros = known.new_zeros(len(known), 4, 24)
        observed = torch.cat([zeros, known], dim=1)
        mask = torch.cat([zeros, torch.ones_like(known)], dim=1)
        return observed, mask

    def predict_noise(self, current, observed, mask, steps):
        noisy_input = torch.cat([current, observed[:, 4:]], dim=1)
        return self.denoiser((noisy_input, observed, mask, steps.reshape(-1, 1)))[:, :4]

    def forward(self, target, conditions, year):
        if target.shape != (len(conditions), 4, 24):
            raise ValueError("Expected target [B,4,24]")
        observed, mask = self.pack_conditions(conditions, year)
        steps = torch.randint(self.num_steps, (len(target),), device=target.device)
        noise = torch.randn_like(target)
        abar = self.alpha_bar[steps, None, None]
        current = abar.sqrt() * target + (1 - abar).sqrt() * noise
        return F.mse_loss(self.predict_noise(current, observed, mask, steps), noise)

    @torch.no_grad()
    def sample(self, conditions, year, generator):
        if self.training:
            raise RuntimeError("Call model.eval() before sampling")
        observed, mask = self.pack_conditions(conditions, year)
        current = torch.randn(len(conditions), 4, 24, device=conditions.device,
                              dtype=conditions.dtype, generator=generator)
        for t in range(self.num_steps - 1, -1, -1):
            steps = torch.full((len(current),), t, device=current.device, dtype=torch.long)
            predicted = self.predict_noise(current, observed, mask, steps)
            current = (current - self.beta[t] / (1 - self.alpha_bar[t]).sqrt() * predicted) / self.alpha[t].sqrt()
            if t > 0:
                current += self.sigma[t] * torch.randn(current.shape, device=current.device,
                                                       dtype=current.dtype, generator=generator)
        return current

    @contextmanager
    def inference_kernels(self):
        """Cache only weight-dependent length-24 kernels within one sampling call.

        Cleared even on failure; no stale kernels after optimization/load/to.
        The diffusion trajectory, S4 weights and RNG are unchanged.
        """
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("Kernel caching requires no-grad eval")
        modules = [m for m in self.modules() if isinstance(m, HippoSSKernel)]
        if any(getattr(m, "_inference_kernel", None) is not None for m in modules):
            raise RuntimeError("Nested inference kernel cache is not supported")
        try:
            for module in modules:
                module._inference_kernel = module(L=24)
            yield
        finally:
            for module in modules:
                module._inference_kernel = None


@torch.no_grad()
def generate_scenarios(model, conditions, year, scenarios, generator, chunk_size=128, progress=None):
    if scenarios < 1 or chunk_size < 1 or len(conditions) < 1:
        raise ValueError("Scenario/chunk/day counts must be positive")
    batches = []
    total = len(conditions) * scenarios
    with model.inference_kernels():
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            indices = torch.arange(start, end, device=conditions.device) // scenarios
            batches.append(model.sample(conditions[indices], year[indices], generator))
            if progress:
                progress(end, total)
    return torch.cat(batches).reshape(len(conditions), scenarios, 4, 24)
