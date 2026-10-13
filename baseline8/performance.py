"""Execution-only logging and thread controls; no changes to model randomness."""
import json
import os
import time
import torch


def configure_threads():
    count = int(os.environ.get("SSSD_CPU_THREADS", "1"))
    if count < 1:
        raise ValueError("SSSD_CPU_THREADS must be positive")
    torch.set_num_threads(count)


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def execution_info(model):
    parameter = next(model.parameters())
    return {"actual_model_device": str(parameter.device),
            "gpu_name": torch.cuda.get_device_name(parameter.device) if parameter.is_cuda else None,
            "torch_threads": torch.get_num_threads(), "sampling_implementation": "SSSD_S4_DDPM_torch_cauchy_cached"}


class Progress:
    def __init__(self, event, **context):
        self.event, self.context = event, context
        self.started = self.last = time.perf_counter()

    def emit(self, force=False, **fields):
        now = time.perf_counter()
        if force or now - self.last >= 30:
            print(json.dumps({"event": self.event, **self.context, **fields,
                              "elapsed_seconds": round(now - self.started, 3)}), flush=True)
            self.last = now
