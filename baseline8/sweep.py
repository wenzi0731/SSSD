from __future__ import annotations

import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from copy import deepcopy
import csv
import json
import re

import yaml

from baseline8.runtime import load_config, resolve_path
from baseline8.train import config_hash, data_manifest, make_dataset, train


def trial_configs(base, search):
    items = search.get("configurations", [])
    if len(items) != 6:
        raise ValueError("The search budget must contain exactly six configurations")
    ids, fingerprints, result = set(), set(), []
    for item in items:
        name = item["id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in ids:
            raise ValueError("Trial IDs must be unique safe directory names")
        config = deepcopy(base)
        for key, value in item["overrides"].items():
            if key not in {"model.res_channels", "training.learning_rate"}:
                raise ValueError(f"Not part of this predeclared search: {key}")
            section, field = key.split(".")
            config[section][field] = value
        fingerprint = config_hash(config)
        if fingerprint in fingerprints:
            raise ValueError("Duplicate effective configurations are not six unique trials")
        ids.add(name)
        fingerprints.add(fingerprint)
        result.append((name, config))
    return result


def sweep(base, search, summary_dir, skip_completed=False):
    configurations = trial_configs(base, search)
    summary_dir = resolve_path(summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    # A completed run cannot be reused after CSV contents/statistics change.
    manifest = data_manifest(base, make_dataset(base, "train"))
    protocol = {"base_config": base, "search_space": search, "data_manifest": manifest}
    protocol_file = summary_dir / "protocol.yaml"
    if protocol_file.exists() and yaml.safe_load(protocol_file.read_text()) != protocol:
        raise ValueError("Existing sweep uses a different protocol; choose a new output directory")
    protocol_file.write_text(yaml.safe_dump(protocol, sort_keys=False))
    rows = []
    for name, config in configurations:
        run_name = f"sssd_tune_{name}"
        run_dir = resolve_path(config["run"]["output_root"]) / f"{run_name}_seed{config['run']['seed']}"
        completed = run_dir / "completed.json"
        if skip_completed and completed.exists():
            if json.loads(completed.read_text())["config_hash"] != config_hash(config):
                raise ValueError(f"Configuration mismatch for completed trial {name}")
            if not (run_dir / "best.pt").exists():
                raise FileNotFoundError(f"Missing checkpoint for {name}")
            if json.loads((run_dir / "data_manifest.json").read_text()) != manifest:
                raise ValueError(f"Training data differs for completed trial {name}")
        else:
            train(config, run_name)
        metrics = json.loads((run_dir / "best_metrics.json").read_text())
        rows.append({"config_id": name, "res_channels": config["model"]["res_channels"],
                     "learning_rate": config["training"]["learning_rate"], **metrics,
                     "checkpoint": str(run_dir / "best.pt")})
        with (summary_dir / "partial_results.json").open("w") as handle:
            json.dump(rows, handle, indent=2)
    rows.sort(key=lambda row: (row["val_macro_nCRPS"], row["config_id"]))
    with (summary_dir / "tuning_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = rows[0]
    best_config = deepcopy(dict(configurations)[best["config_id"]])
    best_config["run"]["name"] = "sssd_final"
    (summary_dir / "best_config.yaml").write_text(yaml.safe_dump(best_config, sort_keys=False))
    (summary_dir / "best_config.json").write_text(json.dumps(best, indent=2))
    print(json.dumps(best, indent=2))
    return best


def main():
    parser = argparse.ArgumentParser(description="Exactly six validation trials; no test evaluation")
    parser.add_argument("--config", default="baseline8/configs/heew.yaml")
    parser.add_argument("--search-space", default="baseline8/configs/search_space.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--output-dir", default="experiments/baseline8/tuning_budget_6")
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()
    base = load_config(args.config, args.overrides)
    base["run"]["seed"] = args.seed
    search = yaml.safe_load(resolve_path(args.search_space).read_text())
    sweep(base, search, args.output_dir, args.skip_completed)


if __name__ == "__main__":
    main()
