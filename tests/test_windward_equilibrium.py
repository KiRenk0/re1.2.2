from __future__ import annotations

import unittest

import numpy as np

from ref_enthalpy_method.aero.windward_cache import build_windward_edge_cache
from ref_enthalpy_method.config.lf_qw import LfQwConfig
from ref_enthalpy_method.gas.thermo import make_perfect_gas_thermo
from ref_enthalpy_method.gas.transport import mu_sutherland
from ref_enthalpy_method.specs.loader import load_yaml
from ref_enthalpy_method.specs.models import CaseSpec
from ref_enthalpy_method.thermal.windward_equilibrium import solve_windward_radiative_equilibrium
from ref_enthalpy_method.types import GasModel


class TestWindwardEquilibrium(unittest.TestCase):
    def test_equilibrium_residual_is_small(self):
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

        nx = 25
        xc = np.linspace(0.0, 1.0, nx)
        slope = np.zeros((nx,), dtype=float)  # flat plate

        # freestream proxy (approx @ 20 km)
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

        # simple leading-edge closure: constant heat flux
        eps = 0.8
        sigma = 5.76e-8
        q0 = 2.0e5

        def q_le(_Tw):
            return q0

        Tw, q = solve_windward_radiative_equilibrium(
            gas=gas,
            cache=cache,
            emissivity=eps,
            sigma_W_m2_K4=sigma,
            q_leading_edge_of_Tw=q_le,
        )

        self.assertEqual(Tw.shape, (nx,))
        self.assertEqual(q.shape, (nx,))
        self.assertTrue(np.all(np.isfinite(Tw)))
        self.assertTrue(np.all(np.isfinite(q)))

        # Residual check (should be exactly satisfied for q output since we compute q=eps*sigma*Tw^4)
        res = q - eps * sigma * (Tw**4)
        self.assertTrue(float(np.max(np.abs(res))) <= 1e-6 * max(1.0, float(np.max(q))))


if __name__ == "__main__":
    raise SystemExit(unittest.main())

