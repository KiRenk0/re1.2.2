"""US Standard Atmosphere 1976 (simplified) - 0 to 32 km.

This matches the baseline `ref_enthalpy.physics.atmosphere.ussa1976_0_32km` behavior closely,
so spec-driven runs are comparable.
"""

from __future__ import annotations

import math


def ussa1976_0_32km(*, h_m: float, R_gas_J_per_kgK: float) -> tuple[float, float, float]:
    """Return (p_Pa, rho_kg_m3, T_K) for 0-32km."""

    h = float(h_m)
    if not (0.0 <= h <= 42000.0):
        raise ValueError(f"ussa1976_0_32km only supports 0-32km, got h_m={h_m}")

    g0 = 9.80665
    R = float(R_gas_J_per_kgK)

    # constants
    T_SL = 288.15
    P_SL = 101325.0
    L_TROP = -0.0065

    T_STRAT1 = 216.65
    P_11KM = 22632.1

    T_STRAT2_BASE = 216.65
    P_20KM = 5474.89
    L_STRAT2 = 0.001

    if h <= 11000.0:
        T_b = T_SL
        P_b = P_SL
        L = L_TROP
        T = T_b + L * h
        P = P_b * (T / T_b) ** (-g0 / (R * L))
    elif h <= 20000.0:
        T = T_STRAT1
        P_b = P_11KM
        h_b = 11000.0
        P = P_b * math.exp(-g0 / (R * T) * (h - h_b))
    else:
        T_b = T_STRAT2_BASE
        P_b = P_20KM
        L = L_STRAT2
        h_b = 20000.0
        T = T_b + L * (h - h_b)
        P = P_b * (T / T_b) ** (-g0 / (R * L))

    rho = P / (R * T)
    return float(P), float(rho), float(T)

