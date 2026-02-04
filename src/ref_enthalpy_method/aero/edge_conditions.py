"""Boundary-layer edge (external) conditions derived from the doc.

Equations covered:
- (2.48) pe/p_inf
- (2.49) pc/p_inf (leading edge reference pressure)
- (2.50) rhoc/rho_inf (Rankine-Hugoniot for normal shock proxy)
- (2.51)-(2.52) rhoe/rho_inf
- (2.53) Te
- (2.54) Mae
- (2.55)-(2.56) effective alpha and Mach with sweep/alpha
"""

from __future__ import annotations

import math

from ..types import EdgeConditions, GasModel


def effective_alpha(alpha_rad: float, chi_w_rad: float) -> float:
    """Eq. (2.55): effective angle of attack with sweep."""

    return math.atan(math.tan(float(alpha_rad)) / math.cos(float(chi_w_rad)))


def effective_ma_inf(ma_inf: float, *, alpha_rad: float, chi_w_rad: float) -> float:
    """Eq. (2.56): effective Mach number with sweep and AoA."""

    ma = float(ma_inf)
    chi = float(chi_w_rad)
    a = float(alpha_rad)
    factor = 1.0 - (math.sin(chi) ** 2) * (math.cos(a) ** 2)
    return ma * math.sqrt(max(factor, 0.0))


def compute_edge_conditions(
    *,
    gas: GasModel,
    ma_inf: float,
    p_inf: float,
    T_inf: float,
    rho_inf: float,
    cp_pressure: float,
    cp0_pressure: float,
) -> EdgeConditions:
    """Compute edge conditions at a windward surface location.

    This follows the sequence in 2.3.3.6, using perfect-gas relations.
    """

    k = float(gas.gamma)
    R = float(gas.R)

    ma = float(ma_inf)
    p_inf = float(p_inf)
    T_inf = float(T_inf)
    rho_inf = float(rho_inf)

    # (2.48) pe/p_inf
    pe_over_pinf = 1.0 + (k / 2.0) * (ma**2) * float(cp_pressure)
    p_e = pe_over_pinf * p_inf

    # (2.49) pc/p_inf at leading edge (using Cp0)
    pc_over_pinf = 1.0 + (k / 2.0) * (ma**2) * float(cp0_pressure)
    p_c = pc_over_pinf * p_inf

    # (2.50) rhoc/rho_inf
    rhoc_over_rhoinf = (6.0 * pc_over_pinf + 1.0) / (pc_over_pinf + 6.0)

    # (2.51) rhoe/rhoc = (pe/pc)^(1/k)
    rhoe_over_rhoc = (p_e / p_c) ** (1.0 / k)

    # (2.52) rhoe/rho_inf
    rhoe_over_rhoinf = rhoe_over_rhoc * rhoc_over_rhoinf
    rho_e = rhoe_over_rhoinf * rho_inf

    # (2.53) Te
    T_e = T_inf * (p_e / p_inf) * (rho_inf / rho_e)

    # (2.54) Mae^2 = 5[ (T_inf/Te)(1+0.2 Ma_inf^2) - 1 ]
    ma_e_sq = 5.0 * ((T_inf / T_e) * (1.0 + 0.2 * ma**2) - 1.0)
    ma_e = math.sqrt(max(ma_e_sq, 0.0))

    a_e = math.sqrt(k * R * T_e)
    v_e = ma_e * a_e
    mu_e = float(gas.mu(T_e))

    return EdgeConditions(p_e=p_e, rho_e=rho_e, T_e=T_e, ma_e=ma_e, a_e=a_e, v_e=v_e, mu_e=mu_e)

