"""Leeward steady radiative-equilibrium coupling.

This solves, pointwise:
    q_leeward(Tw) = eps*sigma*Tw^4

with leeward heat-flux model:
    q = rho_inf * v_inf * St(x) * (h_s - h_w(Tw))
"""

from __future__ import annotations

import numpy as np

from ..types import GasModel
from .wall_temperature import solve_radiative_equilibrium


def solve_leeward_radiative_equilibrium_coupled(
    *,
    gas: GasModel,
    rho_inf: float,
    v_inf: float,
    St_dist: np.ndarray,
    h_s: float,
    emissivity: float,
    sigma_W_m2_K4: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (Tw_dist, q_dist) for leeward surface."""

    St_dist = np.asarray(St_dist, dtype=float).reshape(-1)
    Tw = np.full_like(St_dist, np.nan, dtype=float)
    q = np.full_like(St_dist, np.nan, dtype=float)

    for i in range(St_dist.size):
        St_i = float(St_dist[i])
        if not np.isfinite(St_i):
            continue

        def q_of_Tw(Tw_i: float) -> float:
            h_w = float(gas.h_from_T(float(Tw_i)))
            return float(rho_inf) * float(v_inf) * float(St_i) * (float(h_s) - h_w)

        Tw_i = solve_radiative_equilibrium(
            q_of_Tw=q_of_Tw,
            emissivity=float(emissivity),
            sigma_W_m2_K4=float(sigma_W_m2_K4),
        )
        Tw[i] = Tw_i
        q[i] = float(emissivity) * float(sigma_W_m2_K4) * (float(Tw_i) ** 4) if np.isfinite(Tw_i) else np.nan

    return Tw, q

