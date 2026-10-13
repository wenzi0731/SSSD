"""Rescore existing physical SSSD scenarios; no training or sampling."""
from __future__ import annotations

import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import numpy as np

from baseline8.data import TARGET_COLUMNS
from baseline8.reporting import require_empty_output, write_report
from baseline8.runtime import load_checkpoint, resolve_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--outdir", "--output-dir", required=True)
    parser.add_argument("--metrics-json", help="Optional legacy global_metrics.json; otherwise read it next to the NPZ")
    parser.add_argument("--checkpoint", help="Optional: validate training statistics and recover metric settings")
    parser.add_argument("--seed", type=int, help="Metric subsampling/plot seed, not a new sampling seed")
    parser.add_argument("--precision-recall-k", type=int, default=None)
    parser.add_argument("--max-precision-samples", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    source = resolve_path(args.npz)
    destination = require_empty_output(resolve_path(args.outdir))
    with np.load(source, allow_pickle=False) as payload:
        if not {"scenarios", "targets", "dates"}.issubset(payload.files):
            raise ValueError("Archive requires scenarios, targets and dates")
        samples, targets = payload["scenarios"], payload["targets"]
        dates = payload["dates"].astype(str).tolist()
        labels = payload["channel_names"].astype(str).tolist() if "channel_names" in payload.files else list(TARGET_COLUMNS)
        if labels != list(TARGET_COLUMNS):
            raise ValueError("Expected Electricity, Heat, Cooling, PV channel order")
        archive_seed = int(payload["seed"].item()) if "seed" in payload.files else None
        metadata = json.loads(str(payload["metadata_json"].item())) if "metadata_json" in payload.files else {}
        if not isinstance(metadata, dict):
            raise ValueError("Archive metadata_json must be an object")
        recorded_sample_seed = int(payload["sample_seed"].item()) if "sample_seed" in payload.files else metadata.get("sample_seed")
        if "sampler" in payload.files:
            metadata["sampler"] = str(payload["sampler"].item())
        if {"target_mean", "target_std"}.issubset(payload.files):
            mean, std = payload["target_mean"], payload["target_std"]
            normalization_source = "training statistics stored in source NPZ"
        elif "target_mean" in payload.files or "target_std" in payload.files:
            raise ValueError("Archive must contain both target_mean and target_std")
        else:
            mean = std = None

    if samples.ndim != 4 or samples.shape[2:] != (4, 24) or min(samples.shape) < 1:
        raise ValueError("Require nonempty physical scenarios [day, scenario, 4, 24]")
    if targets.shape != (len(samples), 4, 24) or len(dates) != len(samples):
        raise ValueError("Targets and dates must match the scenario archive")

    # Accept externally exported archives with seed recorded only in a sidecar.
    sidecar = resolve_path(args.metrics_json) if args.metrics_json else source.parent / "global_metrics.json"
    if args.metrics_json or (archive_seed is None and metadata.get("seed") is None and sidecar.exists()):
        sidecar_data = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(sidecar_data, dict):
            raise ValueError("Metrics JSON must be an object")
        for key, expected in (("num_test_days", len(targets)), ("scenarios_per_day", samples.shape[1])):
            if key in sidecar_data and sidecar_data[key] != expected:
                raise ValueError(f"Metrics JSON {key} does not match archive")
        if archive_seed is not None and sidecar_data.get("seed", archive_seed) != archive_seed:
            raise ValueError("Metrics JSON seed conflicts with archive seed")
        metadata = {**sidecar_data, **metadata}
        metadata["source_metrics_json"] = str(sidecar)
    if archive_seed is None and metadata.get("seed") is not None:
        archive_seed = int(metadata["seed"])
    if recorded_sample_seed is None:
        recorded_sample_seed = metadata.get("sample_seed")
    if recorded_sample_seed is None and archive_seed is not None and metadata.get("sampler") == "SSSD_S4_DDPM_full_steps":
        recorded_sample_seed = archive_seed + 90_000

    checkpoint = load_checkpoint(args.checkpoint) if args.checkpoint else None
    if checkpoint is not None and mean is not None:
        manifest = checkpoint.get("data_manifest", {})
        training_stats = manifest.get("normalization", {})
        if manifest.get("target_columns") != labels or not {"target_mean", "target_std"}.issubset(training_stats):
            raise ValueError("Checkpoint lacks matching channel order/training normalization")
        if not (np.array_equal(mean.reshape(-1), np.asarray(training_stats["target_mean"]))
                and np.array_equal(std.reshape(-1), np.asarray(training_stats["target_std"]))):
            raise ValueError("Archive normalization differs from checkpoint")

    if mean is None:
        if not args.checkpoint:
            parser.error("Archive lacks training statistics: supply the matching --checkpoint. Never normalize with test data.")
        manifest = checkpoint.get("data_manifest", {})
        normalization = manifest.get("normalization", {})
        if not {"target_mean", "target_std"}.issubset(normalization):
            raise ValueError("Checkpoint lacks training normalization; cannot safely rescore")
        if manifest.get("target_columns") != labels:
            raise ValueError("Checkpoint channel order differs from archive")
        mean = np.asarray(normalization["target_mean"])
        std = np.asarray(normalization["target_std"])
        normalization_source = "training statistics from matching checkpoint data_manifest"
        metadata.update(checkpoint=str(resolve_path(args.checkpoint)),
                        checkpoint_epoch=checkpoint.get("epoch"))

    if args.seed is None and archive_seed is None:
        parser.error("Cannot recover metric/plot seed: keep the original global_metrics.json next to the NPZ, supply --metrics-json, or explicitly set --seed. Do not infer it from directory names.")
    seed = args.seed if args.seed is not None else archive_seed
    checkpoint_eval = (checkpoint or {}).get("config", {}).get("evaluation", {})
    k = args.precision_recall_k if args.precision_recall_k is not None else int(
        metadata.get("precision_recall_k", checkpoint_eval.get("precision_recall_k", 5)))
    cap = args.max_precision_samples if args.max_precision_samples is not None else int(
        metadata.get("max_precision_samples", checkpoint_eval.get("max_precision_samples", 10_000)))
    metadata.update(
        method="conditional-sssd-baseline8", split="test-2022",
        source_npz=str(source), archive=str(source), rescored_without_sampling=True,
        sample_seed=recorded_sample_seed, original_evaluation_seed=archive_seed,
        normalization_source=normalization_source,
    )
    result = write_report(
        samples, targets, mean, std, dates, labels, destination,
        seed=seed, no_plots=args.no_plots, metadata=metadata,
        precision_recall_k=k,
        max_precision_samples=cap,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
