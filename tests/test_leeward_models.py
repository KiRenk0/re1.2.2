from __future__ import annotations

import unittest

import numpy as np

from ref_enthalpy_method.gas.thermo import make_perfect_gas_thermo
from ref_enthalpy_method.gas.transport import mu_sutherland
from ref_enthalpy_method.heatflux.leeward import (
    leeward_heat_flux_distribution,
    leeward_re_ns,
    leeward_stanton_distribution,
    normal_shock_temperature_ratio,
)
from ref_enthalpy_method.thermal.leeward_equilibrium import solve_leeward_radiative_equilibrium_coupled
from ref_enthalpy_method.types import GasModel


class TestLeewardModels(unittest.TestCase):
    def test_leeward_distribution_finite(self):
        thermo = make_perfect_gas_thermo(cp_const=1005.0)
        gas = GasModel(
            gamma=1.4,
            R=287.0,
            cp_gas=thermo.cp,
            h_from_T=thermo.h_from_T,
            T_from_h=thermo.T_from_h,
            mu=mu_sutherland,
            prandtl=0.72,
        )

        mach = 5.0
        gamma = 1.4
        T_inf = 216.65
        rho_inf = 0.088
        v_inf = 1475.0
        rn_m = 0.02

        ratio_T = normal_shock_temperature_ratio(gamma=gamma, mach=mach)
        T_ns = T_inf * ratio_T
        mu_ns = float(gas.mu(T_ns))
        Re_ns = leeward_re_ns(rho_inf=rho_inf, v_inf=v_inf, R_ref=rn_m, mu_ns=mu_ns)

        nx = 101
        h_s = float(gas.h_from_T(T_inf)) + 0.5 * v_inf**2
        h_w = float(gas.h_from_T(300.0))
        h_wwd_dist = np.full((nx,), float(gas.h_from_T(1000.0)), dtype=float)

        St = leeward_stanton_distribution(Re_ns=Re_ns, h_wwd_dist=h_wwd_dist, h_s=h_s)
        q = leeward_heat_flux_distribution(rho_inf=rho_inf, v_inf=v_inf, St_dist=St, h_s=h_s, h_w=h_w)

        self.assertEqual(q.shape, (nx,))
        self.assertTrue(np.all(np.isfinite(q)))

    def test_leeward_radiative_equilibrium_residual_small(self):
        thermo = make_perfect_gas_thermo(cp_const=1005.0)
        gas = GasModel(
            gamma=1.4,
            R=287.0,
            cp_gas=thermo.cp,
            h_from_T=thermo.h_from_T,
            T_from_h=thermo.T_from_h,
            mu=mu_sutherland,
            prandtl=0.72,
        )

        rho_inf = 0.088
        v_inf = 1475.0
        nx = 5
        St = np.full((nx,), 0.002, dtype=float)
        h_s = float(gas.h_from_T(216.65)) + 0.5 * v_inf**2
        eps = 0.8
        sigma = 5.76e-8

        Tw, q = solve_leeward_radiative_equilibrium_coupled(
            gas=gas, rho_inf=rho_inf, v_inf=v_inf, St_dist=St, h_s=h_s, emissivity=eps, sigma_W_m2_K4=sigma
        )

        self.assertEqual(Tw.shape, (nx,))
        self.assertTrue(np.all(np.isfinite(Tw)))
        self.assertTrue(np.all(np.isfinite(q)))

        # Check equilibrium residual
        res = q - eps * sigma * (Tw**4)
        self.assertTrue(float(np.max(np.abs(res))) <= 1e-3 * max(1.0, float(np.max(q))))


if __name__ == "__main__":
    raise SystemExit(unittest.main())

