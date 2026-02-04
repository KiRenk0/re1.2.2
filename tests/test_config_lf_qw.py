from __future__ import annotations

import unittest

from ref_enthalpy_method.config.lf_qw import LfQwConfig
from ref_enthalpy_method.specs.loader import load_yaml
from ref_enthalpy_method.specs.models import CaseSpec


class TestLfQwConfig(unittest.TestCase):
    def test_defaults(self):
        # minimal case dict
        root = load_yaml("specs/cases/phase1_ma_alpha_h20km.yaml")
        case = CaseSpec.from_yaml_dict(root)
        cfg = LfQwConfig.from_case(case)
        self.assertTrue(cfg.phi_clamp.phi_min_rad > 0)
        self.assertIn(cfg.stagnation.rn_unit, {"cm", "m"})
        self.assertTrue(cfg.stagnation.sweep_exponent_n > 0)
        self.assertIn(cfg.transition.weighting, {"logistic", "step", "hard"})


if __name__ == "__main__":
    raise SystemExit(unittest.main())

