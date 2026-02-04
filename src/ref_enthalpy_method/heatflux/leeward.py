"""Leeward surface mean heat flux density.

Equations covered:
- (2.43) q = rho_inf * v_inf * St * (h_s - h_w)
- (2.44) St correlation
- (2.45) Re_ns (normal-shock Reynolds)
"""

from __future__ import annotations

import numpy as np


def reynolds_ns(
    *,
    rho_inf: float,
    v_inf: float,
    R_ref: float,
    mu_inf: float,
    mu_ns: float,
) -> float:
    """Eq. (2.45) (as written in the doc)."""

    return float(rho_inf) * float(v_inf) * float(R_ref) / float(mu_inf) * (float(mu_inf) / float(mu_ns))


def stanton_number(*, Re_ns: float, h_wwd: float, h_s: float) -> float:
    """Eq. (2.44)."""

    Re_ns = float(Re_ns)
    if not np.isfinite(Re_ns) or Re_ns <= 0.0:
        return float("nan")
    return 0.00282 * (0.7905 + 1.067 * (float(h_wwd) / float(h_s))) * (Re_ns ** -0.37)


def leeward_mean_heat_flux(
    *,
    rho_inf: float,
    v_inf: float,
    St: float,
    h_s: float,
    h_w: float,
) -> float:
    """Eq. (2.43)."""

    return float(rho_inf) * float(v_inf) * float(St) * (float(h_s) - float(h_w))


def normal_shock_temperature_ratio(*, gamma: float, mach: float) -> float:
    """Baseline proxy used to estimate T_ns/T_inf for mu_ns in Re_ns.

    Not from the doc excerpt directly; this is the engineering correlation used in the baseline code.
    """

    g = float(gamma)
    M = float(mach)
    return ((2 * g * M**2 - (g - 1)) * ((g - 1) * M**2 + 2)) / ((g + 1) ** 2 * M**2)


def leeward_re_ns(
    *,
    rho_inf: float,
    v_inf: float,
    R_ref: float,
    mu_ns: float,
) -> float:
    """Convenience: Re_ns = rho_inf * v_inf * R_ref / mu_ns."""

    return float(rho_inf) * float(v_inf) * float(R_ref) / float(mu_ns)


def leeward_stanton_distribution(*, Re_ns: float, h_wwd_dist: np.ndarray, h_s: float) -> np.ndarray:
    """Compute St(x) pointwise from (2.44) using pointwise h_wwd(x)."""

    h_wwd_dist = np.asarray(h_wwd_dist, dtype=float).reshape(-1)
    St = np.full_like(h_wwd_dist, np.nan, dtype=float)
    for i in range(h_wwd_dist.size):
        h_wwd = float(h_wwd_dist[i])
        if not np.isfinite(h_wwd):
            continue
        St[i] = stanton_number(Re_ns=float(Re_ns), h_wwd=h_wwd, h_s=float(h_s))
    return St


def leeward_heat_flux_distribution(
    *,
    rho_inf: float,
    v_inf: float,
    St_dist: np.ndarray,
    h_s: float,
    h_w: float,
) -> np.ndarray:
    """Compute q(x) pointwise from (2.43) given St(x)."""

    St_dist = np.asarray(St_dist, dtype=float).reshape(-1)
    q = np.full_like(St_dist, np.nan, dtype=float)
    for i in range(St_dist.size):
        St = float(St_dist[i])
        if not np.isfinite(St):
            continue
        q[i] = leeward_mean_heat_flux(rho_inf=rho_inf, v_inf=v_inf, St=St, h_s=h_s, h_w=h_w)
    return q

