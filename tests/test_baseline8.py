from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from baseline8.data import BASE_WEATHER_COLUMNS, PV_WEATHER_COLUMNS, TARGET_COLUMNS
from baseline8.model import ConditionalSSSD, generate_scenarios
from baseline8.runtime import load_config, load_checkpoint
from baseline8.sweep import sweep, trial_configs
from baseline8.train import make_dataset, train, validate


@pytest.fixture
def tiny_config(tmp_path):
    torch.set_num_threads(1)
    config = load_config("baseline8/configs/heew.yaml")
    config["model"].update(res_channels=4, skip_channels=4, num_res_layers=1, s4_d_state=4,
                           diffusion_step_embed_dim_in=8, diffusion_step_embed_dim_mid=8,
                           diffusion_step_embed_dim_out=8)
    config["diffusion"]["num_steps"] = 3
    config["training"].update(epochs=1, batch_size=2, val_every=1)
    config["evaluation"].update(batch_size=2, val_scenarios=2, test_scenarios=3, sample_chunk_size=4)
    config["run"].update(device="cpu", output_root=str(tmp_path / "runs"))
    timestamps = pd.DatetimeIndex([ts for year in (2014, 2020, 2021, 2022)
                                  for ts in pd.date_range(f"{year}-01-01", periods=48, freq="h")])
    cols = dict(Year=timestamps.year, Month=timestamps.month, Day=timestamps.day, Hour=timestamps.hour)
    rng = np.random.default_rng(12)
    energy, weather = pd.DataFrame(cols), pd.DataFrame(cols)
    for i, label in enumerate(TARGET_COLUMNS):
        energy[label] = rng.uniform(10, 50, len(timestamps)) * (i + 1)
    for label in BASE_WEATHER_COLUMNS + PV_WEATHER_COLUMNS:
        weather[label] = rng.uniform(0, 1, len(timestamps))
    for kind, frame in (("energy", energy), ("weather", weather)):
        path = tmp_path / f"{kind}.csv"
        frame.to_csv(path, index=False)
        config["data"][f"{kind}_path"] = str(path)
    return config


def test_reference_metric_source_is_exact():
    content = Path("baseline8/metrics.py").read_bytes()
    blob = b"blob " + str(len(content)).encode() + b"\0" + content
    assert hashlib.sha1(blob).hexdigest() == "ea17ab63806f45539fc4dce630115b1717e1ad55"


def test_imputer_architecture_matches_upstream():
    import ast
    def body(path):
        tree = ast.parse(Path(path).read_text())
        tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        return ast.dump(tree)
    assert body("src/imputers/SSSDS4Imputer.py") == body("baseline8/vendor/imputer.py")


def test_cauchy_conjugates_and_gradients():
    from baseline8.vendor.s4 import cauchy_slow
    v = torch.randn(2, 3, dtype=torch.cdouble, requires_grad=True)
    w = torch.randn(2, 3, dtype=torch.cdouble) - 3
    z = torch.randn(5, dtype=torch.cdouble) + 3
    actual = cauchy_slow(v, z, w)
    expected = sum(v[:, n, None] / (z - w[:, n, None]) +
                   v[:, n, None].conj() / (z - w[:, n, None].conj()) for n in range(3))
    torch.testing.assert_close(actual, expected)
    actual.abs().square().sum().backward()
    assert torch.isfinite(v.grad).all() and v.grad.abs().sum() > 0


@pytest.mark.parametrize("state_size", [4, 64])
def test_s4_fft_kernel_matches_dense_state_recurrence(state_size):
    from baseline8.vendor.s4 import HippoSSKernel
    kernel = HippoSSKernel(H=2, N=state_size, L=24, precision=2)
    actual = kernel(L=24)
    kernel.kernel.setup_step("dense")
    state = kernel.default_state(1)
    values = []
    for t in range(24):
        impulse = torch.full((1, 2), float(t == 0), dtype=torch.float64)
        value, state = kernel.kernel.step(impulse, state)
        values.append(value.real)
    torch.testing.assert_close(actual.double(), torch.stack(values, -1)[0], rtol=1e-5, atol=1e-6)


def test_clean_known_channels_and_ddpm_equation(tiny_config):
    model = ConditionalSSSD(tiny_config, 18).eval()
    calls = []
    class Spy(torch.nn.Module):
        def forward(self, inputs):
            noisy, observed, mask, steps = inputs
            assert torch.count_nonzero(mask[:, :4]) == 0
            assert torch.count_nonzero(observed[:, :4]) == 0
            assert (mask[:, 4:] == 1).all()
            torch.testing.assert_close(noisy[:, 4:], observed[:, 4:])
            calls.append(steps[0].item())
            out = torch.full_like(noisy, float("nan"))
            out[:, :4] = .25
            return out
    model.denoiser = Spy()
    c, y = torch.randn(2, 18, 24), torch.randn(2, 1, 24)
    actual = model.sample(c, y, torch.Generator().manual_seed(5))
    rng = torch.Generator().manual_seed(5)
    expected = torch.randn(2, 4, 24, generator=rng)
    for t in range(model.num_steps - 1, -1, -1):
        expected = (expected - model.beta[t] / (1 - model.alpha_bar[t]).sqrt() * .25) / model.alpha[t].sqrt()
        if t:
            expected += model.sigma[t] * torch.randn(expected.shape, generator=rng)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert calls == [2, 1, 0]
    # NaN known-lane outputs are excluded from both training loss and sampling.
    assert torch.isfinite(model(torch.randn(2, 4, 24), c, y))


def test_sampling_cache_rng_and_joint_gradient(tiny_config):
    from baseline8.vendor.s4 import HippoSSKernel
    model = ConditionalSSSD(tiny_config, 18)
    c, y = torch.randn(2, 18, 24), torch.randn(2, 1, 24)
    # Upstream final projection starts at zero; first update unlocks gradients.
    loss = model(torch.randn(2, 4, 24), c, y)
    loss.backward()
    torch.optim.Adam(model.parameters(), lr=.01).step()
    model.eval()
    state = torch.random.get_rng_state()
    def draw(cond=c, seed=42):
        return generate_scenarios(model, cond, y, 3, torch.Generator().manual_seed(seed), 4)
    a = draw()
    assert a.shape == (2, 3, 4, 24) and torch.isfinite(a).all()
    torch.testing.assert_close(a, draw(), rtol=0, atol=0)
    assert not torch.allclose(a, draw(seed=43))
    assert not torch.allclose(a, draw(cond=c + 1))
    assert torch.equal(state, torch.random.get_rng_state())
    # Uncached evaluation uses exactly the same day/scenario order and draws.
    rng = torch.Generator().manual_seed(42)
    uncached = []
    for start, end in ((0, 4), (4, 6)):
        indices = torch.arange(start, end) // 3
        uncached.append(model.sample(c[indices], y[indices], rng))
    torch.testing.assert_close(a, torch.cat(uncached).reshape(2, 3, 4, 24), rtol=0, atol=0)
    modules = [m for m in model.modules() if isinstance(m, HippoSSKernel)]
    with pytest.raises(ValueError, match="intentional"):
        with torch.no_grad(), model.inference_kernels():
            raise ValueError("intentional")
    assert all(m._inference_kernel is None for m in modules)
    current = torch.randn(2, 4, 24, requires_grad=True)
    cond = c.clone().requires_grad_()
    observed, mask = model.pack_conditions(cond, y)
    model.predict_noise(current, observed, mask, torch.ones(2, dtype=torch.long))[:, 0].sum().backward()
    assert (current.grad.abs().sum((0, 2)) > 0).all()
    assert (cond.grad.abs().sum((0, 2)) > 0).all()
    model.train()
    with pytest.raises(RuntimeError, match="eval"):
        draw()


def test_default_diffusion_schedule(tiny_config):
    tiny_config["diffusion"]["num_steps"] = 200
    model = ConditionalSSSD(tiny_config, 18).eval()
    torch.testing.assert_close(model.beta, torch.linspace(.0001, .02, 200))
    assert torch.isfinite(model.sigma).all() and model.alpha_bar[-1] > 0
    samples = generate_scenarios(model, torch.randn(1, 18, 24), torch.zeros(1, 1, 24),
                                 1, torch.Generator().manual_seed(1), 1)
    assert torch.isfinite(samples).all()


def test_validation_does_not_consume_training_rng(tiny_config):
    from torch.utils.data import DataLoader
    data = make_dataset(tiny_config, "val")
    loader = DataLoader(data, batch_size=2, generator=torch.Generator().manual_seed(88))
    model = ConditionalSSSD(tiny_config, 18)
    state = torch.random.get_rng_state()
    a = validate(model, loader, data, tiny_config, torch.device("cpu"))
    assert torch.equal(state, torch.random.get_rng_state())
    b = validate(model, loader, data, tiny_config, torch.device("cpu"))
    assert a == b


def test_train_only_scaling_and_splits(tiny_config):
    train_data = make_dataset(tiny_config, "train")
    val_data, test_data = make_dataset(tiny_config, "val"), make_dataset(tiny_config, "test")
    assert len(train_data) == 4 and len(val_data) == len(test_data) == 2
    assert all(d.startswith("2021") for d in val_data.dates)
    assert all(d.startswith("2022") for d in test_data.dates)
    frame = pd.read_csv(tiny_config["data"]["energy_path"])
    expected = frame.loc[frame.Year <= 2020, TARGET_COLUMNS].to_numpy(np.float32)
    np.testing.assert_allclose(train_data.target_mean, expected.mean(0), rtol=1e-6)
    np.testing.assert_array_equal(train_data.target_mean, test_data.target_mean)
    np.testing.assert_array_equal(train_data.target_std, val_data.target_std)
    frame.loc[frame.Year == 2022, TARGET_COLUMNS] += 10000
    frame.to_csv(tiny_config["data"]["energy_path"], index=False)
    changed = make_dataset(tiny_config, "test")
    np.testing.assert_array_equal(changed.target_mean, train_data.target_mean)
    torch.testing.assert_close(changed[0][0], test_data[0][0])


@pytest.mark.parametrize("issue", ["nan", "duplicate", "incomplete"])
def test_bad_data(tiny_config, issue):
    path = tiny_config["data"]["weather_path"]
    frame = pd.read_csv(path)
    if issue == "nan":
        frame.loc[0, "Humidity"] = np.nan
    elif issue == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    else:
        frame = frame.drop(frame.index[0])
    frame.to_csv(path, index=False)
    if issue == "incomplete":
        assert len(make_dataset(tiny_config, "train")) == 3
    else:
        with pytest.raises(ValueError):
            make_dataset(tiny_config, "train")


def test_six_trials_and_reject_duplicates(tiny_config):
    search = yaml.safe_load(Path("baseline8/configs/search_space.yaml").read_text())
    assert len(trial_configs(tiny_config, search)) == 6
    bad = deepcopy(search)
    bad["configurations"].pop()
    with pytest.raises(ValueError, match="six"):
        trial_configs(tiny_config, bad)
    bad = deepcopy(search)
    bad["configurations"][1]["overrides"] = bad["configurations"][0]["overrides"]
    with pytest.raises(ValueError, match="Duplicate"):
        trial_configs(tiny_config, bad)
    with pytest.raises(ValueError, match="Unknown"):
        load_config("baseline8/configs/heew.yaml", ["diffusion.nonexistent=100"])


def test_checkpoint_to_outputs_and_rescore(tiny_config, tmp_path, monkeypatch):
    checkpoint = train(tiny_config, "smoke")
    payload = load_checkpoint(checkpoint)
    assert payload["method"] == "conditional-sssd-baseline8"
    history = [json.loads(line) for line in (checkpoint.parent / "history.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in history] == [1]
    from baseline8.evaluate import main
    out = tmp_path / "evaluation"
    argv = ["evaluate", "--checkpoint", str(checkpoint), "--outdir", str(out), "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    with np.load(out / "baseline8_scenarios.npz", allow_pickle=False) as archive:
        assert archive["scenarios"].shape == (2, 3, 4, 24)
        assert (archive["scenarios"][:, :, 3] >= 0).all()
        assert archive["dates"].tolist() == make_dataset(tiny_config, "test").dates
    metrics = json.loads((out / "global_metrics.json").read_text())
    for label in TARGET_COLUMNS:
        for suffix in ("RMSE_Z", "MAE_Z", "Precision_Z", "Recall_Z", "CR", "IW", "R2", "IS", "IS_Z"):
            assert np.isfinite(metrics[f"{label}_{suffix}"])
    assert metrics["sample_seed"] == 90042 and metrics["training_seed"] == 42
    assert metrics["VS_pair_count"] == 3456
    assert all(np.isfinite(metrics[k]) for k in ("ES", "VS", "ES_Z", "VS_Z"))
    assert len(pd.read_csv(out / "channel_metrics.csv")) == 4
    assert len(pd.read_csv(out / "daily_scores.csv")) == 2
    assert (out / "metric_definitions.json").exists()
    assert len(list((out / "random_timeseries_50").glob("*.png"))) == 2
    assert (out / "pearson" / "generated_global_pearson.png").exists()
    assert (out / "global_pearson.png").exists()
    from baseline8.score_npz import main as score_main
    rescored = tmp_path / "rescored"
    monkeypatch.setattr(sys, "argv", ["score_npz", "--npz", str(out / "baseline8_scenarios.npz"),
                                      "--outdir", str(rescored), "--no-plots"])
    score_main()
    result = json.loads((rescored / "global_metrics.json").read_text())
    for key in ("ES", "VS", "Electricity_R2", "PV_IS", "Heat_IS_Z"):
        assert result[key] == metrics[key]
    with pytest.raises(FileExistsError):
        train(tiny_config, "smoke")
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(FileExistsError):
        main()
    frame = pd.read_csv(tiny_config["data"]["energy_path"])
    frame.loc[0, "PV"] += 1
    frame.to_csv(tiny_config["data"]["energy_path"], index=False)
    monkeypatch.setattr(sys, "argv", argv[:-4] + ["--outdir", str(tmp_path / "mismatch"), "--device", "cpu"])
    with pytest.raises(ValueError, match="differs"):
        main()


def test_sweep_and_usable_winner(tiny_config, tmp_path):
    search = {"configurations": [{"id": f"trial{i}", "overrides": {
        "model.res_channels": 4, "training.learning_rate": (i + 1) * 1e-4}} for i in range(6)]}
    out = tmp_path / "sweep"
    best = sweep(tiny_config, search, out)
    winner = yaml.safe_load((out / "best_config.yaml").read_text())
    assert winner["training"]["learning_rate"] == best["learning_rate"]
    assert len(pd.read_csv(out / "tuning_results.csv")) == 6
    assert sweep(tiny_config, search, out, skip_completed=True) == best
    winner["run"]["seed"] = 123
    assert train(winner).exists()
    bad = deepcopy(tiny_config)
    bad["training"]["epochs"] = 2
    with pytest.raises(ValueError, match="protocol"):
        sweep(bad, search, out, skip_completed=True)


@pytest.mark.parametrize("entry", ["train", "evaluate", "sweep", "score_npz"])
def test_cli_direct_and_module(entry):
    for args in (["-m", f"baseline8.{entry}"], [f"baseline8/{entry}.py"]):
        result = subprocess.run([sys.executable, *args, "--help"], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
