"""Transport properties (viscosity etc.)."""

from __future__ import annotations

import math


def mu_sutherland(T: float, *, mu0: float = 1.716e-5, T0: float = 273.15, S: float = 110.4) -> float:
    """Sutherland law for dynamic viscosity of air (Pa*s).

    mu = mu0 * (T/T0)^(3/2) * (T0 + S)/(T + S)
    """

    T = float(T)
    return mu0 * (T / T0) ** 1.5 * (T0 + S) / (T + S)

