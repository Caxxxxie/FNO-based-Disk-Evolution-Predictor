import argparse
import unittest

from scripts.generate_fargo_dataset_v1 import (
    ASPECT_RATIO_MAX,
    ASPECT_RATIO_MIN,
    SMOKE_CASES,
    cases_from_args,
)


class GenerateFargoDatasetTest(unittest.TestCase):
    def test_default_aspect_ratios_stay_in_ppdonet_domain(self):
        self.assertTrue(
            all(ASPECT_RATIO_MIN <= case.aspect_ratio <= ASPECT_RATIO_MAX for case in SMOKE_CASES)
        )

    def test_random_aspect_ratios_stay_in_ppdonet_domain(self):
        args = argparse.Namespace(num_random_cases=32, seed=0)
        cases = cases_from_args(args)
        self.assertTrue(
            all(ASPECT_RATIO_MIN <= case.aspect_ratio <= ASPECT_RATIO_MAX for case in cases)
        )


if __name__ == "__main__":
    unittest.main()
