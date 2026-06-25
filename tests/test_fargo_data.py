from pathlib import Path
import unittest

import numpy as np

from fargo_data import FargoMemmapDataset, make_rollout_batch


ROOT = Path(__file__).resolve().parents[1]


class FargoDataTest(unittest.TestCase):
    def test_rollout_batch_shapes(self):
        ds = FargoMemmapDataset(
            ROOT / "data" / "v3_smoke4f6_64x32",
            ["log_sigma", "delta_v_r", "delta_v_theta"],
        )
        rng = np.random.default_rng(0)
        mean = np.zeros(ds.n_channels, dtype=np.float32)
        std = np.ones(ds.n_channels, dtype=np.float32)
        batch = make_rollout_batch(ds, rng, ds.train_cases, horizon=2, batch_size=2, mean=mean, std=std)
        self.assertEqual(batch["x0"].shape, (2, ds.ny, ds.nx, ds.n_channels))
        self.assertEqual(batch["targets"].shape, (2, 2, ds.ny, ds.nx, ds.n_channels))
        self.assertEqual(batch["mu"].shape, (2, 3))
        self.assertEqual(batch["t"].shape, (2, 2, 1))
        self.assertEqual(batch["dt"].shape, (2, 2, 1))

    def test_rollout_batch_rejects_too_long_horizon(self):
        ds = FargoMemmapDataset(
            ROOT / "data" / "v3_smoke4f6_64x32",
            ["log_sigma", "delta_v_r", "delta_v_theta"],
        )
        rng = np.random.default_rng(0)
        mean = np.zeros(ds.n_channels, dtype=np.float32)
        std = np.ones(ds.n_channels, dtype=np.float32)
        with self.assertRaises(ValueError):
            make_rollout_batch(ds, rng, ds.train_cases, horizon=ds.n_frames, batch_size=2, mean=mean, std=std)


if __name__ == "__main__":
    unittest.main()
