from __future__ import annotations

import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from copy import deepcopy
import hashlib
import json
import platform
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from baseline8.data import HEEWConditionDataset
from baseline8.metrics import crps_map
from baseline8.model import ConditionalSSSD, generate_scenarios
from baseline8.runtime import choose_device, load_config, resolve_path, seed_everything, seed_worker
UPSTREAM_COMMIT = "4d3b7a51c54b658945c0ba0bbb26e5ee1f763bed"
from baseline8.performance import Progress, configure_threads, execution_info, synchronize


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def data_manifest(config, dataset):
    result = {"normalization": {}, "files": {}}
    for key in ("target_mean", "target_std", "weather_mean", "weather_std"):
        result["normalization"][key] = getattr(dataset, key).tolist()
    result["target_columns"] = dataset.target_cols
    result["weather_columns"] = dataset.weather_cols
    for key in ("energy_path", "weather_path"):
        digest = hashlib.sha256()
        with resolve_path(config["data"][key]).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result["files"][key] = digest.hexdigest()
    return result


def make_dataset(config, split):
    data = config["data"]
    return HEEWConditionDataset(
        resolve_path(data["energy_path"]), resolve_path(data["weather_path"]),
        split=split, weather_feature_set=data["weather_feature_set"],
    )


@torch.no_grad()
def validate(model, loader, dataset, config, device):
    model.eval()
    progress = Progress("validation")
    progress.emit(force=True, phase="start", batches=len(loader),
                  days=len(dataset), **execution_info(model))
    rng = torch.Generator(device=device).manual_seed(int(config["run"]["seed"]) + 50000)
    scores = np.zeros(4, dtype=np.float64)
    denominators = np.zeros(4, dtype=np.float64)
    for batch_index, (conditions, pv_year, target, _) in enumerate(loader, 1):
        samples = generate_scenarios(
            model, conditions.to(device), pv_year.to(device),
            int(config["evaluation"]["val_scenarios"]), rng,
            int(config["evaluation"]["sample_chunk_size"]),
            progress=lambda done, total, hour=None: progress.emit(
                phase="sampling", batch=batch_index, batches=len(loader),
                chunk_scenarios_done=done, batch_scenarios=total, hour=hour),
        )
        if not torch.isfinite(samples).all():
            raise FloatingPointError("Nonfinite validation scenarios")
        physical = dataset.denormalize(samples).cpu().numpy()
        physical[:, :, 3] = np.maximum(physical[:, :, 3], 0)
        truth = dataset.denormalize(target).numpy()
        scores += crps_map(physical, truth).sum(axis=(0, 2))
        denominators += np.abs(truth).sum(axis=(0, 2))
        progress.emit(force=batch_index in (1, len(loader)), phase="batch_done",
                      batch=batch_index, batches=len(loader))
    result = {"val_macro_nCRPS": float((scores / (denominators + 1e-8)).mean())}
    result.update({f"val_{label}_nCRPS": float(scores[i] / (denominators[i] + 1e-8))
                   for i, label in enumerate(dataset.target_cols)})
    return result


def train(config, run_name=None):
    configure_threads()
    seed = int(config["run"]["seed"])
    seed_everything(seed)
    if config["data"].get("seq_len", 24) != 24:
        raise ValueError("baseline8 requires 24-hour days")
    device = choose_device(config["run"]["device"])
    name = run_name or config["run"]["name"]
    run_dir = resolve_path(config["run"]["output_root"]) / f"{name}_seed{seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {run_dir}. Use a new --run-name.")
    training = config["training"]
    if min(int(training[k]) for k in ("epochs", "batch_size", "val_every", "early_stopping_patience")) < 1:
        raise ValueError("Training counts must be positive")
    train_data = make_dataset(config, "train")
    val_data = make_dataset(config, "val")
    manifest = data_manifest(config, train_data)
    train_loader = DataLoader(
        train_data, batch_size=int(training["batch_size"]), shuffle=True,
        num_workers=int(config["data"]["num_workers"]), worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_data, batch_size=int(config["evaluation"]["batch_size"]), shuffle=False,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    model = ConditionalSSSD(config, train_data.condition_channels).to(device)
    decay = float(training["ema_decay"])
    if decay != 0:
        raise ValueError("baseline8 uses raw weights; EMA must remain 0 for upstream weight_norm modules")
    ema = model
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training["learning_rate"]), betas=tuple(training["betas"])
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (run_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    (run_dir / "split_dates.json").write_text(json.dumps({
        "train": train_data.dates, "validation": val_data.dates,
    }, indent=2))
    environment = {"python": platform.python_version(), "torch": str(torch.__version__),
                   "device": str(device), "parameters": sum(p.numel() for p in model.parameters()),
                   "upstream_commit": UPSTREAM_COMMIT,
                   "conditioning_protocol": "weather_time_year_only_no_energy_history"}
    environment.update(execution_info(model))
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2))
    print(json.dumps({"event": "run_start", "run_dir": str(run_dir),
                      "train_days": len(train_data), "validation_days": len(val_data),
                      **environment}), flush=True)
    best_score, stale = float("inf"), 0
    best_path = run_dir / "best.pt"
    started = time.perf_counter()
    for epoch in range(1, int(training["epochs"]) + 1):
        synchronize(device)
        epoch_started = time.perf_counter()
        progress = Progress("training", epoch=epoch)
        progress.emit(force=True, phase="start", batches=len(train_loader))
        model.train()
        loss_sum, seen = 0.0, 0
        for batch_index, (conditions, pv_year, target, _) in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)
            loss = model(target.to(device), conditions.to(device), pv_year.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss in epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]),
                                          error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.item() * len(target)
            seen += len(target)
            progress.emit(phase="batch_done", batch=batch_index, batches=len(train_loader))
        synchronize(device)
        training_seconds = time.perf_counter() - epoch_started
        progress.emit(force=True, phase="done", batches=len(train_loader))
        record = {"epoch": epoch, "train_loss": loss_sum / seen,
                  "training_seconds": training_seconds}
        if epoch % int(training["val_every"]) == 0 or epoch == int(training["epochs"]):
            validation_started = time.perf_counter()
            metrics = validate(ema, val_loader, val_data, config, device)
            synchronize(device)
            record["validation_seconds"] = time.perf_counter() - validation_started
            record.update(metrics)
            score = metrics["val_macro_nCRPS"]
            if not np.isfinite(score):
                raise FloatingPointError("Nonfinite validation score")
            if score < best_score:
                best_score, stale = score, 0
                payload = {
                    "format_version": 1, "method": "conditional-sssd-baseline8",
                    "config": deepcopy(config), "epoch": epoch,
                    "model": ema.state_dict(), "raw_model": model.state_dict(),
                    "metrics": metrics, "data_manifest": manifest,
                    "condition_channels": train_data.condition_channels,
                    "upstream_commit": UPSTREAM_COMMIT,
                    "conditioning_protocol": "weather_time_year_only_no_energy_history",
                }
                torch.save(payload, best_path)
                (run_dir / "best_metrics.json").write_text(json.dumps({"epoch": epoch, **metrics}, indent=2))
            else:
                stale += 1
        record["epoch_seconds"] = time.perf_counter() - epoch_started
        with (run_dir / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if stale >= int(training["early_stopping_patience"]):
            break
    if not best_path.exists():
        raise RuntimeError("No validated checkpoint was saved")
    (run_dir / "completed.json").write_text(json.dumps({
        "config_hash": config_hash(config), "training_seconds": time.perf_counter() - started,
        "best_checkpoint": str(best_path), "best_val_macro_nCRPS": best_score,
    }, indent=2))
    return best_path


def main():
    parser = argparse.ArgumentParser(description="Train condition-only joint SSSD")
    parser.add_argument("--config", default="baseline8/configs/heew.yaml")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--run-name")
    args = parser.parse_args()
    print(train(load_config(args.config, args.overrides), args.run_name))


if __name__ == "__main__":
    main()
