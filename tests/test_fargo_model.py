import unittest

import jax.numpy as jnp
import numpy as np

from fargo_model import (
    FargoFNOConfig,
    FargoPointwiseConfig,
    make_fargo_fno,
    make_fargo_pointwise,
)


class FargoModelTest(unittest.TestCase):
    def test_pointwise_output_shape_matches_state(self):
        coords = np.zeros((4, 6, 3), dtype=np.float32)
        model = make_fargo_pointwise(coords, FargoPointwiseConfig(width=8, depth=2, output_channels=3))
        batch = {
            "x": jnp.zeros((2, 4, 6, 3), dtype=jnp.float32),
            "mu": jnp.zeros((2, 3), dtype=jnp.float32),
            "t": jnp.zeros((2, 1), dtype=jnp.float32),
            "dt": jnp.ones((2, 1), dtype=jnp.float32),
        }
        params = model.init(jnp.asarray([0, 1], dtype=jnp.uint32), batch)
        pred = model.apply(params, batch)
        self.assertEqual(pred.shape, batch["x"].shape)

    def test_fno_output_shape_matches_state(self):
        coords = np.zeros((4, 6, 3), dtype=np.float32)
        model = make_fargo_fno(
            coords,
            FargoFNOConfig(
                width=8,
                depth=1,
                modes_theta=3,
                output_channels=3,
                radial_kernels=(3,),
                radial_dilations=(1,),
            ),
        )
        batch = {
            "x": jnp.zeros((2, 4, 6, 3), dtype=jnp.float32),
            "mu": jnp.zeros((2, 3), dtype=jnp.float32),
            "t": jnp.zeros((2, 1), dtype=jnp.float32),
            "dt": jnp.ones((2, 1), dtype=jnp.float32),
        }
        params = model.init(jnp.asarray([0, 2], dtype=jnp.uint32), batch)
        pred = model.apply(params, batch)
        self.assertEqual(pred.shape, batch["x"].shape)


if __name__ == "__main__":
    unittest.main()
