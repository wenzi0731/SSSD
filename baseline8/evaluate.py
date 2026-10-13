from __future__ import annotations

import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline8.data import HEEWConditionDataset
from baseline8.reporting import require_empty_output, write_report
from baseline8.model import ConditionalSSSD, generate_scenarios
from baseline8.runtime import choose_device, load_checkpoint, resolve_path, seed_everything
from baseline8.train import data_manifest
from baseline8.performance import Progress, configure_threads, execution_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate conditional joint SSSD.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--energy-path", default=None)
    parser.add_argument("--weather-path", default=None)
    parser.add_argument("--outdir", "--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scenarios", "--num-scenarios", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-days", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    configure_threads()
    args = parse_args()
    checkpoint_path = resolve_path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.get("method") != "conditional-sssd-baseline8":
        raise ValueError("Expected a baseline8 SSSD checkpoint")
    config = checkpoint["config"]
    seed = int(args.seed if args.seed is not None else config["run"]["seed"])
    seed_everything(seed)
    args.scenarios = int(config["evaluation"]["test_scenarios"]) if args.scenarios is None else args.scenarios
    args.batch_size = int(config["evaluation"]["batch_size"]) if args.batch_size is None else args.batch_size
    if args.scenarios < 1 or args.batch_size < 1 or (args.max_days is not None and args.max_days < 1):
        raise ValueError("Counts must be positive")
    device = choose_device(args.device)
    data_config = config["data"]
    dataset = HEEWConditionDataset(
        energy_path=resolve_path(args.energy_path or data_config["energy_path"]),
        weather_path=resolve_path(args.weather_path or data_config["weather_path"]),
        split="test",
        weather_feature_set=data_config.get("weather_feature_set", "pv10"),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    manifest_config = json.loads(json.dumps(config))
    manifest_config["data"]["energy_path"] = str(args.energy_path or data_config["energy_path"])
    manifest_config["data"]["weather_path"] = str(args.weather_path or data_config["weather_path"])
    if data_manifest(manifest_config, dataset) != checkpoint["data_manifest"]:
        raise ValueError("Data or normalization differs from training checkpoint")
    generator = ConditionalSSSD(config, dataset.condition_channels)
    generator.load_state_dict(checkpoint["model"])
    generator.to(device).eval()
    output_dir = (
        resolve_path(args.outdir)
        if args.outdir
        else checkpoint_path.parent / "evaluation_test"
    )
    require_empty_output(output_dir)
    progress = Progress("test_sampling")
    progress.emit(force=True, phase="start", days=min(args.max_days or len(dataset), len(dataset)),
                  scenarios=args.scenarios, **execution_info(generator))

    generated_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    condition_batches: list[np.ndarray] = []
    year_batches: list[np.ndarray] = []
    dates: list[str] = []
    noise_generator = torch.Generator(device=device).manual_seed(seed + 90_000)
    elapsed = 0.0
    processed = 0
    with torch.no_grad():
        for conditions, pv_year, target, batch_dates in loader:
            if args.max_days is not None and processed >= args.max_days:
                break
            take = len(batch_dates)
            if args.max_days is not None:
                take = min(take, args.max_days - processed)
            conditions = conditions[:take].to(device)
            pv_year = pv_year[:take].to(device)
            target = target[:take].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            generated_z = generate_scenarios(
                generator,
                conditions,
                pv_year,
                args.scenarios,
                noise_generator,
                int(config["evaluation"]["sample_chunk_size"]),
                progress=lambda done, total, hour=None: progress.emit(
                    phase="sampling", completed_days=processed,
                    chunk_scenarios_done=done, batch_scenarios=total, hour=hour),
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - start
            if not torch.isfinite(generated_z).all():
                raise FloatingPointError("Nonfinite generated scenarios before denormalization")
            generated = dataset.denormalize(generated_z)
            target_physical = dataset.denormalize(target)
            generated[:, :, 3].clamp_(min=0.0)
            generated_batches.append(generated.cpu().numpy())
            target_batches.append(target_physical.cpu().numpy())
            condition_batches.append(conditions.cpu().numpy())
            year_batches.append(pv_year.cpu().numpy())
            dates.extend(list(batch_dates[:take]))
            processed += take
            progress.emit(phase="batch_done", completed_days=processed)
    progress.emit(force=True, phase="done", completed_days=processed)

    generated_values = np.concatenate(generated_batches)
    target_values = np.concatenate(target_batches)
    condition_values = np.concatenate(condition_batches)
    labels = list(dataset.target_cols)
    archive = output_dir / "baseline8_scenarios.npz"
    metrics = write_report(
        generated_values, target_values, dataset.target_mean, dataset.target_std,
        dates, labels, output_dir, seed=seed, no_plots=args.no_plots,
        precision_recall_k=int(config["evaluation"]["precision_recall_k"]),
        max_precision_samples=int(config["evaluation"]["max_precision_samples"]),
        metadata={
            **execution_info(generator),
            "method": "conditional-sssd-baseline8",
            "upstream_commit": checkpoint.get("upstream_commit"),
            "training_timesteps": generator.num_steps,
            "sampling_steps": generator.num_steps,
            "condition_columns": dataset.weather_cols + dataset.time_cols + ["year_index"],
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "split": "test-2022",
            "partial_test": len(target_values) != len(dataset),
            "training_seed": int(config["run"]["seed"]),
            "sample_seed": seed + 90_000,
            "sampler": "SSSD_S4_DDPM_full_steps",
            "conditioning_protocol": "weather_time_year_only_no_energy_history",
            "sampling_chunk_size": int(config["evaluation"]["sample_chunk_size"]),
            "sampling_batch_size": args.batch_size,
            "sampling_seconds": elapsed,
            "seconds_per_1000_scenarios": elapsed * 1000 / (len(target_values) * args.scenarios),
            "archive": str(archive),
        },
    )
    np.savez_compressed(
        archive,
        scenarios=generated_values,
        targets=target_values,
        conditions=condition_values,
        pv_year=np.concatenate(year_batches),
        dates=np.asarray(dates),
        channel_names=np.asarray(labels),
        target_mean=dataset.target_mean,
        target_std=dataset.target_std,
        seed=seed,
        sample_seed=seed + 90_000,
        sampler="SSSD_S4_DDPM_full_steps",
        num_scenarios=args.scenarios,
        metadata_json=json.dumps(metrics, allow_nan=False),
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
