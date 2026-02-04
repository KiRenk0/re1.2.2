from __future__ import annotations

import unittest

from ref_enthalpy_method.heatflux.leading_edge import leading_edge_heat_flux_baseline


class TestLeadingEdge(unittest.TestCase):
    def test_leading_edge_q_decreases_with_radius(self):
        # Keep everything else fixed; larger radius -> lower heat flux (RN^-0.5)
        common = dict(
            c_root_m=2.0,
            chord_m=2.0,
            rn_unit="cm",
            sweep_exponent_n=1.5,
            rho_inf=0.088,
            v_inf=1475.0,
            h0=1.0e6,
            h_w=3.0e5,
            h_300K=3.0e5,
            chi_w_rad=0.0,
            alpha_rad=0.0,
        )
        q_small = leading_edge_heat_flux_baseline(rn_le_m=0.01, **common)
        q_large = leading_edge_heat_flux_baseline(rn_le_m=0.02, **common)
        self.assertTrue(q_small > q_large)


if __name__ == "__main__":
    raise SystemExit(unittest.main())

