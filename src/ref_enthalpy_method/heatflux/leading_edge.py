"""Leading-edge heat flux density formulas.

Equations covered:
- (2.39) q_sl = q_sph / sqrt(2)
- (2.40) modified Kemp-Riddell for q_sph
- (2.41) sweep/AoA correction
"""

from __future__ import annotations

import math

from ..constants import RHO_SL, V_C


def kemp_riddell_modified_qsph(
    *,
    R_N: float,
    rho_inf: float,
    v_inf: float,
    h0: float,
    h_w: float,
    h_300K: float,
) -> float:
    """Eq. (2.40) sphere stagnation-line heat flux density (W/m^2)."""

    RN = float(R_N)
    if RN <= 0:
        raise ValueError("R_N must be positive.")

    return (
        110311.7
        * (RN ** -0.5)
        * ((float(rho_inf) / RHO_SL) ** 0.5)
        * ((float(v_inf) / V_C) ** 3.15)
        * ((float(h0) - float(h_w)) / (float(h0) - float(h_300K)))
    )


def leading_edge_heat_flux(
    *,
    q0: float,
    chi_w_rad: float,
    alpha_rad: float,
    n: float = 1.5,
) -> float:
    """Eq. (2.41) correction for sweep & AoA, using non-swept/no-AoA q0 as normalization."""

    chi = float(chi_w_rad)
    a = float(alpha_rad)
    base = 1.0 - (math.sin(chi) ** 2) * (math.cos(a) ** 2)
    return float(q0) * (max(base, 0.0) ** (float(n) / 2.0))


def leading_edge_cylinder_stagnation_qsl(*, q_sph: float) -> float:
    """Eq. (2.39): q_sl = q_sph / sqrt(2)."""

    return float(q_sph) / math.sqrt(2.0)


def leading_edge_heat_flux_baseline(
    *,
    rn_le_m: float,
    c_root_m: float,
    chord_m: float,
    rn_unit: str,
    sweep_exponent_n: float,
    rho_inf: float,
    v_inf: float,
    h0: float,
    h_w: float,
    h_300K: float,
    chi_w_rad: float,
    alpha_rad: float,
) -> float:
    """Baseline-style leading-edge heat flux (W/m^2).

    - Uses rn_local = rn_le*(chord/c_root)
    - Uses rn_unit in {"cm","m"} to decide whether to convert rn_local (m -> cm)
    - Uses Eq (2.40) as implemented in baseline: compute q in W/cm^2 then *1e4 to W/m^2
    - Applies Eq (2.39) and Eq (2.41)
    """

    rn_local_m = float(rn_le_m) * float(chord_m) / float(c_root_m)
    unit = str(rn_unit).strip().lower()
    rn_val = rn_local_m * 100.0 if unit == "cm" else rn_local_m
    rn_val = max(float(rn_val), 1e-6)

    # Eq (2.40) but in baseline numeric convention: q_w_cm2 then convert to SI.
    term1 = 110311.7 / math.sqrt(rn_val)
    term2 = math.sqrt(float(rho_inf) / float(RHO_SL))
    term3 = (float(v_inf) / float(V_C)) ** 3.15
    term4 = (float(h0) - float(h_w)) / (float(h0) - float(h_300K))
    q_sph_SI = float(term1 * term2 * term3 * term4) * 10000.0

    q0 = leading_edge_cylinder_stagnation_qsl(q_sph=q_sph_SI)  # Eq (2.39)
    return leading_edge_heat_flux(q0=q0, chi_w_rad=float(chi_w_rad), alpha_rad=float(alpha_rad), n=float(sweep_exponent_n))

