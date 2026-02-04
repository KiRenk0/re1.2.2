"""Wall temperature solvers from heat balance equations.

Equations covered:
- Radiation: q_r = eps * sigma * T_w^4  (2.33)
- Boost phase balance: q_a - q_r = rho * c * delta * dT_w/dt  (2.57)
- Cruise phase balance: q_a = q_r  (2.58)

Note: in the doc, wall enthalpy is substituted as h_w = c_p * T_w (c_p is gas cp, not Cp pressure coefficient).
"""

from __future__ import annotations

import math

from ..constants import SIGMA_BOLTZMANN
from ..types import Material, RunOptions


def q_radiation(*, emissivity: float, T_w: float) -> float:
    return float(emissivity) * SIGMA_BOLTZMANN * float(T_w) ** 4


def step_wall_temperature_boost(
    *,
    material: Material,
    q_a: float,
    T_w: float,
    dt: float,
) -> float:
    """One explicit Euler step for (2.57)."""

    q_r = q_radiation(emissivity=material.emissivity, T_w=T_w)
    dTdt = (float(q_a) - q_r) / (float(material.rho) * float(material.c) * float(material.delta))
    return float(T_w) + float(dt) * dTdt


def solve_wall_temperature_cruise(
    *,
    emissivity: float,
    alpha_h: float,
    h_r: float,
    cp_gas: float,
    options: RunOptions = RunOptions(),
    T_init: float = 500.0,
) -> float:
    """Solve cruise balance (2.58) by fixed-point / damped Newton.

    Solve: alpha_h * (h_r - cp_gas*T) = eps*sigma*T^4
    """

    eps = float(emissivity)
    ah = float(alpha_h)
    hr = float(h_r)
    cp = float(cp_gas)

    T = max(float(T_init), 1.0)
    for _ in range(int(options.max_iter)):
        f = ah * (hr - cp * T) - eps * SIGMA_BOLTZMANN * T**4
        if abs(f) < options.tol:
            return T

        # derivative w.r.t T
        df = -ah * cp - eps * SIGMA_BOLTZMANN * 4.0 * T**3
        if df == 0.0:
            break

        # damped Newton step
        step = f / df
        T_next = T - step
        if not math.isfinite(T_next) or T_next <= 0:
            T_next = max(T * 0.5, 1.0)

        # mild damping to avoid overshoot
        T = 0.7 * T + 0.3 * T_next

    return T


def solve_radiative_equilibrium(
    *,
    q_of_Tw,
    emissivity: float,
    sigma_W_m2_K4: float = SIGMA_BOLTZMANN,
    T_lo_K: float = 300.0,
    T_hi_K: float = 4000.0,
) -> float:
    """Solve steady radiative equilibrium: q_a(Tw) = eps*sigma*Tw^4 via bracketing/bisection.

    This is typically more robust than Newton for this problem.
    """

    eps = float(emissivity)
    sigma = float(sigma_W_m2_K4)

    def f(T: float) -> float:
        return float(q_of_Tw(T)) - eps * sigma * (float(T) ** 4)

    flo = f(float(T_lo_K))
    fhi = f(float(T_hi_K))

    if flo == 0.0:
        return float(T_lo_K)
    if not math.isfinite(flo):
        return float("nan")

    # If upper bound is NaN, try expanding
    if not math.isfinite(fhi):
        T_hi_try = float(T_hi_K)
        for _ in range(12):
            T_hi_try *= 2.0
            fhi = f(T_hi_try)
            if math.isfinite(fhi):
                T_hi_K = T_hi_try
                break
        if not math.isfinite(fhi):
            return float("nan")

    if fhi == 0.0:
        return float(T_hi_K)

    # Ensure sign change; if not, expand upper bound
    if flo * fhi > 0.0:
        T_hi_try = float(T_hi_K)
        for _ in range(12):
            T_hi_try *= 2.0
            fhi = f(T_hi_try)
            if not math.isfinite(fhi):
                continue
            if flo * fhi <= 0.0:
                T_hi_K = T_hi_try
                break
        if flo * fhi > 0.0:
            # No root in [T_lo, T_hi] even after expansion.
            # Engineering fallback: if f(T_lo) < 0, then q_a(T_lo) < q_r(T_lo),
            # meaning the radiative equilibrium temperature lies below T_lo.
            # In that case, clamp to T_lo instead of returning NaN (keeps fields finite).
            if math.isfinite(flo) and flo < 0.0:
                return float(T_lo_K)
            return float("nan")

    lo, hi = float(T_lo_K), float(T_hi_K)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if not math.isfinite(fm):
            return float("nan")
        if abs(fm) <= 1e-6 * max(1.0, abs(flo), abs(fhi)):
            return float(mid)
        if flo * fm <= 0.0:
            hi = mid
            fhi = fm
        else:
            lo = mid
            flo = fm
    return float(0.5 * (lo + hi))

