"""Numerical/protocol tests; synthetic inputs here are NOT paper results."""
import copy
import json
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.supplementary.data import Errors, Windows, corrupt, failure_mask, fit_scaler
from experiments.supplementary.evaluate import correlations
from experiments.supplementary.fewshot import adapt
from experiments.supplementary.models import Adapter, defaults

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


class ProtocolTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("tables"), "PyTables optional for CSV/NPZ users")
    def test_original_hdf_filenames(self):
        from experiments.supplementary.data import read_data
        with tempfile.TemporaryDirectory(prefix="stgnn-hdf-test-") as td:
            folder = Path(td) / "metr-la"
            folder.mkdir()
            frame = pd.DataFrame(np.ones((30, 12), dtype=np.float32),
                                 index=pd.date_range("2020-01-01", periods=30, freq="5min"))
            frame.to_hdf(folder / "metr-la.h5", key="df")
            values, time, metadata = read_data(td, "metr-la")
            self.assertEqual(values.shape, (30, 12))
            self.assertAlmostEqual(float(time[1]), 5 / (24 * 60))

    def test_failure_assignment_and_covariates(self):
        x = torch.arange(2 * 3 * 10 * 2).reshape(2, 3, 10, 2).float()
        failed = failure_mask(10, .3, 42)
        kept = ~failed
        x[:, :, kept, 0] = 0.  # Legitimate zeros on kept sensors remain kept.
        for kind in ("zero", "stale", "noise", "bias", "intermittent"):
            y = corrupt(x, failed, kind, [0, 1], 42, mean=np.ones(10), std=np.ones(10))
            self.assertTrue(torch.equal(x[:, :, kept], y[:, :, kept]))
            self.assertTrue(torch.equal(x[..., 1], y[..., 1]))
        self.assertEqual(int(failed.sum()), 3)

    def test_random_corruptions_independent_of_batch_size(self):
        x = torch.randn(4, 3, 10, 2)
        failed = failure_mask(10, .3, 42)
        for kind in ("noise", "intermittent"):
            whole = corrupt(x, failed, kind, [0, 1, 2, 3], 7, mean=[1], std=[2])
            pieces = torch.cat([corrupt(x[i:i+1], failed, kind, [i], 7, mean=[1], std=[2]) for i in range(4)])
            self.assertTrue(torch.equal(whole, pieces))

    def test_zero_fill_spaces(self):
        x = torch.ones(1, 2, 4, 2)
        failed = np.array([True, False, False, False])
        physical = corrupt(x, failed, "zero", [0], 1, mean=[10], std=[2], fill_space="physical")
        normalized = corrupt(x, failed, "zero", [0], 1, mean=[10], std=[2])
        self.assertTrue(torch.all(physical[:, :, 0, 0] == -5))
        self.assertTrue(torch.all(normalized[:, :, 0, 0] == 0))

    def test_masked_vs_normalized_metrics_and_weighted_batches(self):
        target = torch.tensor([[[0., 2.]], [[4., 0.]], [[8., 4.]]])
        pred = torch.zeros_like(target)
        expected = Errors([True, False])
        expected.add(pred, target, [0], [2])
        split = Errors([True, False])
        split.add(pred[:2], target[:2], [0], [2])
        split.add(pred[2:], target[2:], [0], [2])
        self.assertEqual(expected.result(), split.result())
        whole = expected.result()["whole"]
        self.assertEqual(whole["normalized_mae"]["count"], 6)
        self.assertEqual(whole["masked_physical_mae"]["count"], 4)
        self.assertAlmostEqual(whole["masked_physical_mae"]["mae"], 4.5)

    def test_train_only_scaling_and_no_boundary_crossing(self):
        values = np.arange(100, dtype=np.float32)[:, None]
        mean, std = fit_scaler(values)
        self.assertEqual(mean.item(), 34.5)
        shifted = values.copy()
        shifted[70:] += 1000
        np.testing.assert_equal(fit_scaler(shifted)[0], mean)
        windows = Windows(values, np.zeros(100), mean, std, "test", 3, 2)
        self.assertEqual(len(windows), 16)
        np.testing.assert_equal(windows[0][1].numpy().ravel(), [83, 84])
        np.testing.assert_equal(windows[15][1].numpy().ravel(), [98, 99])

    def test_fewshot_exact_steps_and_k0_no_updates(self):
        model = Adapter("gru", defaults("gru", 12, 3, 2, 4), torch.eye(12), torch.eye(12), "cpu")
        values = np.random.default_rng(4).normal(size=(100, 12)).astype(np.float32)
        mean, std = fit_scaler(values)
        windows = Windows(values, np.zeros(100), mean, std, "train", 3, 2)
        initial = copy.deepcopy(model.state_dict())
        self.assertEqual(adapt(model, windows, [], steps=30, batch_size=2, lr=.001, seed=42, device="cpu"), [])
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in initial.items()))
        loss = adapt(model, windows, [0, 1], steps=30, batch_size=2, lr=.001, seed=42, device="cpu")
        self.assertEqual(len(loss), 30)
        self.assertTrue(any(not torch.equal(v, model.state_dict()[k]) for k, v in initial.items()))
        adapted = copy.deepcopy(model.state_dict())
        model.load_state_dict(initial)
        repeated = adapt(model, windows, [0, 1], steps=30, batch_size=2, lr=.001, seed=42, device="cpu")
        self.assertEqual(loss, repeated)
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in adapted.items()))

    def test_all_architectures_forward_backward_batch_one(self):
        for name in ("dcrnn", "gru", "stgcn", "mtgnn", "gwnet", "gwnet_strong", "agcrn", "megacrn", "gts", "dgcrn"):
            with self.subTest(model=name):
                features = np.random.default_rng(1).normal(size=(32, 12)).astype(np.float32)
                model = Adapter(name, defaults(name, 12, 3, 2, 4, 32), torch.eye(12), torch.eye(12), "cpu", features)
                model.train()
                x, yt, y = torch.randn(1, 3, 12, 2), torch.rand(1, 2, 12), torch.randn(1, 2, 12)
                pred = model(x, yt, y, 1)
                self.assertEqual(tuple(pred.shape), (1, 2, 12))
                (pred - y).square().mean().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_exact_permutations_and_degenerate_vectors(self):
        result = correlations(range(6), range(6), list("abcdef"))
        self.assertEqual(result["permutations"], 720)
        self.assertAlmostEqual(result["p_exact_two_sided"], 2 / 720)
        self.assertEqual(len(result["leave_one_out"]), 6)
        self.assertIsNone(correlations([1]*6, range(6), list("abcdef"))["pearson"])


class EndToEndTests(unittest.TestCase):
    def test_all_four_table_commands_and_missing_input_failure(self):
        from experiments.supplementary.__main__ import needed_checkpoints
        from experiments.supplementary.train import train
        with tempfile.TemporaryDirectory(prefix="stgnn-supplementary-test-") as td:
            temp = Path(td)
            data = temp / "data"
            rng = np.random.default_rng(3)
            for dataset, n in (("metr-la", 12), ("pems-bay", 14), ("pems04", 13)):
                folder = data / dataset
                folder.mkdir(parents=True)
                values = rng.uniform(1, 20, size=(160, n)).astype(np.float32)
                values[::13, 0] = 0
                if dataset == "pems04":
                    np.savez(folder / "pems04.npz", data=values[..., None])
                else:
                    pd.DataFrame(values, index=pd.date_range("2020-01-01", periods=160, freq="5min")).to_csv(
                        folder / ("METR-LA.csv" if dataset == "metr-la" else "PEMS-BAY.csv"))
                np.save(folder / "adj_mx.npy", rng.uniform(.1, 1, size=(n, n)).astype(np.float32))
            cfg = json.loads((ROOT / "configs/supplementary_experiments.json").read_text(encoding="utf-8"))
            cfg["protocol"].update(seeds=[0], mask_seeds=[42, 43], batch_size=7)
            cfg["training"]["gts_feature_steps"] = 32
            cfg["table21"]["budgets"] = [0, 2, 5]
            for dataset, models in cfg["checkpoints"].items():
                for name in models:
                    models[name] = str(temp / "weights" / f"{name}__{dataset}__seed{{seed}}.pt")
            config = temp / "config.json"
            config.write_text(json.dumps(cfg), encoding="utf-8")
            for dataset, name, seed in needed_checkpoints(cfg):
                train(cfg, data, name, dataset, seed, cfg["checkpoints"][dataset][name].format(seed=seed),
                      "cpu", epochs=1, max_batches=1, hidden=4, seq_len=3, horizon=2)
            for table in ("table10", "table11", "table21", "table31"):
                command = [sys.executable, "-m", "experiments.supplementary", table, "--config", str(config),
                           "--data-root", str(data), "--output", str(temp / table), "--max-windows", "8", "--threads", "1"]
                completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                result = json.loads((temp / table / "results.json").read_text(encoding="utf-8"))
                self.assertTrue(result["complete"])
                self.assertEqual(result["window_limit"], 8)
                if table == "table10":
                    self.assertEqual(len(result["records"]), 5 * 5 * 2)
                elif table == "table11":
                    self.assertEqual(len(result["records"]), 6 * 5 * 5 * 2)
                elif table == "table21":
                    self.assertEqual(len(result["records"]), 4 * 3)
                    for row in result["records"]:
                        self.assertEqual(row["steps"], 0 if row["k"] == 0 else 30)
                else:
                    self.assertEqual(len(result["architecture_measurements"]), 6 * 3)
                    for entry in result["correlations"].values():
                        self.assertEqual(entry["n"], 6)
                # Rerunning cannot silently replace an existing result.
                self.assertNotEqual(subprocess.run(command, cwd=ROOT, capture_output=True).returncode, 0)
            bad = copy.deepcopy(cfg)
            bad["checkpoints"]["metr-la"]["gts"] = str(temp / "absent.pt")
            config.write_text(json.dumps(bad), encoding="utf-8")
            command = [sys.executable, "-m", "experiments.supplementary", "table10", "--config", str(config),
                       "--data-root", str(data), "--output", str(temp / "missing"), "--max-windows", "1"]
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse((temp / "missing/results.json").exists())


if __name__ == "__main__":
    unittest.main()
