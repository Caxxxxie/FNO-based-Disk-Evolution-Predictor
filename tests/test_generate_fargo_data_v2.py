from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.generate_fargo_data_v2 import create_memmaps, infer_completed_cases


class GenerateFargoDataV2Test(unittest.TestCase):
    def test_infer_completed_cases_from_existing_memmap(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp)
            shape = (3, 2, 4, 5)
            arrays = create_memmaps(dataset_dir, shape)
            arrays["log_sigma"][0] = 0.0
            arrays["log_sigma"][1] = 1.0
            arrays["log_sigma"][2] = np.nan
            for array in arrays.values():
                array.flush()

            completed = infer_completed_cases(dataset_dir, shape)

            self.assertEqual(completed.tolist(), [False, True, False])

    def test_resume_rejects_incomplete_memmap_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp)
            np.lib.format.open_memmap(
                dataset_dir / "log_sigma.npy",
                mode="w+",
                dtype=np.float32,
                shape=(1, 2, 3, 4),
            )

            with self.assertRaises(ValueError):
                create_memmaps(dataset_dir, (1, 2, 3, 4), resume=True)


if __name__ == "__main__":
    unittest.main()
