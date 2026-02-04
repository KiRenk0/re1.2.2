from __future__ import annotations

import unittest

import numpy as np

from ref_enthalpy_method.aero.windward_cache import build_windward_edge_cache, windward_q_distribution_from_Tw
from ref_enthalpy_method.config.lf_qw import LfQwConfig
from ref_enthalpy_method.gas.thermo import make_perfect_gas_thermo
from ref_enthalpy_method.gas.transport import mu_sutherland
from ref_enthalpy_method.specs.loader import load_yaml
from ref_enthalpy_method.specs.models import CaseSpec
from ref_enthalpy_method.types import GasModel


class TestWindwardCache(unittest.TestCase):
    def test_cache_and_q_are_finite(self):
        root = load_yaml("specs/cases/phase1_ma_alpha_h20km.yaml")
        case = CaseSpec.from_yaml_dict(root)
        lf_cfg = LfQwConfig.from_case(case)

        thermo = make_perfect_gas_thermo(cp_const=case.cp_J_per_kgK)
        gas = GasModel(
            gamma=case.gamma,
            R=case.R_J_per_kgK,
            cp_gas=thermo.cp,
            h_from_T=thermo.h_from_T,
            T_from_h=thermo.T_from_h,
            mu=mu_sutherland,
            prandtl=case.pr,
        )

        nx = 51
        xc = np.linspace(0.0, 1.0, nx)
        slope = np.zeros((nx,), dtype=float)  # flat plate

        # freestream proxy (consistent with our USSA use at 20km)
        p_inf = 5474.0
        rho_inf = 0.088
        T_inf = 216.65

        cache = build_windward_edge_cache(
            gas=gas,
            lf_cfg=lf_cfg,
            mach=5.0,
            alpha_deg=5.0,
            sweep_le_deg=0.0,
            p_inf=p_inf,
            rho_inf=rho_inf,
            T_inf=T_inf,
            chord_m=2.0,
            xc_grid=xc,
            slope_arr=slope,
            transition_x_over_c=case.transition_x_over_c,
        )

        self.assertEqual(len(cache.edges), nx)
        self.assertEqual(cache.x_over_c.shape, (nx,))
        self.assertEqual(cache.x_phys.shape, (nx,))
        self.assertTrue(np.all(np.isfinite(cache.x_phys)))

        Tw = np.full((nx,), 900.0, dtype=float)
        q = windward_q_distribution_from_Tw(gas=gas, cache=cache, Tw=Tw)
        self.assertEqual(q.shape, (nx,))
        # index 0 left for leading edge in solver
        self.assertTrue(np.all(np.isfinite(q[1:])))


if __name__ == "__main__":
    raise SystemExit(unittest.main())

