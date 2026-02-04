from __future__ import annotations

import math
import unittest

import numpy as np

from ref_enthalpy_method.thermal.transient import march_explicit_balance
from ref_enthalpy_method.thermal.wall_temperature import solve_radiative_equilibrium


class TestWallModels(unittest.TestCase):
    def test_radiative_equilibrium_constant_heat_flux(self):
        eps = 0.8
        sigma = 5.76e-8
        q0 = 2.0e5

        def q_of_Tw(_Tw):
            return q0

        Tw = solve_radiative_equilibrium(q_of_Tw=q_of_Tw, emissivity=eps, sigma_W_m2_K4=sigma)
        self.assertTrue(math.isfinite(Tw))
        Tw_expected = (q0 / (eps * sigma)) ** 0.25
        self.assertTrue(abs(Tw - Tw_expected) / Tw_expected < 1e-3)

    def test_transient_balance_shapes(self):
        n = 5
        Tw0 = np.full((n,), 300.0, dtype=float)
        eps = 0.8
        sigma = 5.76e-8
        q0 = 1.0e5
        cap = 2700.0 * 900.0 * 0.002  # rho*c*delta

        def eval_q_a(_Tw):
            return np.full((n,), q0, dtype=float)

        Tw_time, q_time = march_explicit_balance(
            Tw0=Tw0,
            dt_s=0.01,
            n_steps=3,
            cap_J_per_m2K=cap,
            emissivity=eps,
            sigma_W_m2_K4=sigma,
            Tw_min_K=150.0,
            Tw_max_K=6000.0,
            eval_q_a=eval_q_a,
        )
        self.assertEqual(Tw_time.shape, (4, n))
        self.assertEqual(q_time.shape, (4, n))
        self.assertTrue(np.all(np.isfinite(q_time)))
        self.assertTrue(np.all(np.isfinite(Tw_time)))


if __name__ == "__main__":
    raise SystemExit(unittest.main())

