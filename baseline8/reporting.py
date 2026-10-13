"""Baseline5-compatible outputs without modifying the SSSD sampler."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from baseline8.metrics import save_additional_scores
from baseline8.metrics import (
    summarize, save_global_pearson_comparison, save_global_pearson_plot,
    save_random_timeseries_plots,
)


CHANNEL_SUFFIXES = (
    "MAE", "RMSE", "MAE_Z", "RMSE_Z", "R2", "CRPS", "nCRPS",
    "Precision_Z", "Recall_Z", "CR", "IW", "IS", "IS_Z",
)


def require_empty_output(output_dir):
    destination = Path(output_dir)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"Use a new, empty output directory: {destination}")
    return destination


def write_report(
    samples, target, target_mean, target_std, dates, labels, output_dir,
    *, seed=42, precision_recall_k=5, max_precision_samples=10_000,
    no_plots=False, metadata=None,
):
    destination = require_empty_output(output_dir)
    samples, target = np.asarray(samples), np.asarray(target)
    mean, std = np.asarray(target_mean).reshape(-1), np.asarray(target_std).reshape(-1)
    dates, labels = list(map(str, dates)), list(map(str, labels))
    if samples.ndim != 4 or samples.shape[2:] != (4, 24) or min(samples.shape) < 1:
        raise ValueError("Require nonempty physical scenarios [day, scenario, 4, 24]")
    if target.shape != (samples.shape[0], 4, 24):
        raise ValueError("Targets must have shape [day, 4, 24]")
    if len(dates) != len(target) or len(set(dates)) != len(dates) or len(labels) != 4:
        raise ValueError("Require one unique date per day and four channel labels")
    if any(Path(date).name != date for date in dates):
        raise ValueError("Dates cannot contain path separators")
    if mean.shape != (4,) or std.shape != (4,) or np.any(std <= 0):
        raise ValueError("Require four training means and positive training standard deviations")
    if not all(np.isfinite(value).all() for value in (samples, target, mean, std)):
        raise ValueError("Nonfinite scoring inputs; do not replace NaN/Inf with zeros")
    if precision_recall_k < 1 or max_precision_samples < 2:
        raise ValueError("Require precision_recall_k >= 1 and max_precision_samples >= 2")

    metrics = summarize(samples, target, labels, mean, std, seed,
                        precision_recall_k, max_precision_samples)
    metrics.update(save_additional_scores(
        samples, target, mean, std, dates, labels, destination,
    ))
    random_paths = []
    if not no_plots:
        metrics.update(save_global_pearson_comparison(
            target, samples, destination / "pearson", labels,
        ))
        save_global_pearson_plot(target, destination / "global_pearson.png", labels)
        random_paths = save_random_timeseries_plots(
            target, samples, dates, destination / "random_timeseries_50", labels,
            seed=seed, n_plots=50,
        )

    result = {
        **(metadata or {}), **metrics,
        "num_test_days": len(target), "scenarios_per_day": samples.shape[1],
        "seed": seed, "interval_alpha": 0.05, "VS_p": 0.5,
        "precision_recall_k": precision_recall_k,
        "max_precision_samples": max_precision_samples,
        "metric_reference": "CSDI baseline5 17c217b; cross_channel_unit_weights_v1",
        "global_metrics": str(destination / "global_metrics.csv"),
        "global_pearson": None if no_plots else str(destination / "global_pearson.png"),
        "pearson_dir": None if no_plots else str(destination / "pearson"),
        "random_timeseries_dir": None if no_plots else str(destination / "random_timeseries_50"),
        "random_timeseries_count": len(random_paths),
    }
    result = {
        key: None if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value
        for key, value in result.items()
    }
    with (destination / "channel_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["channel", *CHANNEL_SUFFIXES])
        writer.writeheader()
        for label in labels:
            writer.writerow({"channel": label, **{suffix: result[f"{label}_{suffix}"] for suffix in CHANNEL_SUFFIXES}})
    for name in ("global_metrics.json", "metrics_summary.json"):
        (destination / name).write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    with (destination / "global_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(result.items())
    # Additional one-row layout for combining runs; legacy metric/value stays above.
    with (destination / "global_metrics_wide.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    return result
