"""Windward steady radiative-equilibrium solver (Doc Eq 2.58).

Solves pointwise:
  q_windward(Tw, i) = eps*sigma*Tw^4

The leading-edge term (i=0) is provided by the caller as a q(Tw) closure.
For i>=1, q(Tw) is evaluated from cached edge state.
"""

from __future__ import annotations

import numpy as np

from ..aero.windward_cache import WindwardEdgeCache, windward_q_at_index
from ..types import GasModel
from .wall_temperature import solve_radiative_equilibrium


def solve_windward_radiative_equilibrium(
    *,
    gas: GasModel,
    cache: WindwardEdgeCache,
    emissivity: float,
    sigma_W_m2_K4: float,
    q_leading_edge_of_Tw,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (Tw_dist, q_dist) for windward surface."""

    eps = float(emissivity)
    sigma = float(sigma_W_m2_K4)

    nx = int(cache.x_over_c.size)
    Tw = np.full((nx,), np.nan, dtype=float)
    q = np.full((nx,), np.nan, dtype=float)

    # Leading edge (i=0)
    def q0(Tw0: float) -> float:
        return float(q_leading_edge_of_Tw(float(Tw0)))

    Tw0 = solve_radiative_equilibrium(q_of_Tw=q0, emissivity=eps, sigma_W_m2_K4=sigma)
    Tw[0] = Tw0
    q[0] = eps * sigma * (Tw0**4) if np.isfinite(Tw0) else np.nan

    # i>=1
    for i in range(1, nx):

        def qi(Tw_i: float) -> float:
            return windward_q_at_index(gas=gas, cache=cache, i=i, Tw_i=float(Tw_i))

        Tw_i = solve_radiative_equilibrium(q_of_Tw=qi, emissivity=eps, sigma_W_m2_K4=sigma)
        Tw[i] = Tw_i
        q[i] = eps * sigma * (Tw_i**4) if np.isfinite(Tw_i) else np.nan

    return Tw, q

