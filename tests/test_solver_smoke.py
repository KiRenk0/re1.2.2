from __future__ import annotations

import unittest

import numpy as np


class TestSolverSmoke(unittest.TestCase):
    def test_run_case_smoke_outputs_finite(self):
        from ref_enthalpy_method.solver import WingLowFidelitySolver

        solver = WingLowFidelitySolver(
            vehicle_config="specs/vehicles/doubleconvex_t0p03_sweep0_v0.yaml",
            case_config="specs/cases/phase1_ma_alpha_h20km.yaml",
            sampling_config="specs/sampling/baseline_root_windward_chord_line_101.yaml",
            run_dir="runs/test_smoke",
        )

        _u = solver.compute_snapshot(mach=5.0, alpha=5.0)
        fields = solver.last_fields
        self.assertIn("q_w", fields)
        self.assertIn("q_l", fields)
        self.assertIn("Tw_w", fields)
        self.assertIn("Tw_l", fields)

        for k in ("q_w", "q_l", "Tw_w", "Tw_l"):
            arr = np.asarray(fields[k], dtype=float).reshape(-1)
            self.assertTrue(arr.size > 0)
            self.assertTrue(np.all(np.isfinite(arr)))


if __name__ == "__main__":
    raise SystemExit(unittest.main())

