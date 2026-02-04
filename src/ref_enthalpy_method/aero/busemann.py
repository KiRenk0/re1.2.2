"""Busemann theory pressure coefficient approximation.

Implements eq. (2.47) and its helper coefficients c1/c2/c3 from Ma_inf.
"""

from __future__ import annotations

import math


def _busemann_coeffs(ma_inf: float) -> tuple[float, float, float]:
    ma = float(ma_inf)
    if ma <= 1.0:
        raise ValueError("Busemann approximation expects supersonic/hypersonic Ma_inf > 1.")

    c1 = 2.0 / math.sqrt(ma**2 - 1.0)
    c2 = ((ma**2 - 2.0) ** 2 + 1.4 * ma**4) / ((ma**2 - 1.0) ** 2)
    c3 = (
        (0.36 * ma**8 - 1.493 * ma**6 + 3.6 * ma**4 - 2.0 * ma**2 + 1.33)
        / ((ma**2 - 1.0) ** 3.5)
    )
    return c1, c2, c3


def busemann_cp(*, ma_inf: float, phi_rad: float) -> float:
    """Pressure coefficient Cp from Busemann theory (eq. 2.47).

    Parameters
    - ma_inf: freestream Mach number (use effective Mach if you apply sweep/alpha correction first)
    - phi_rad: local angle between tangent and freestream direction (radians)
    """

    c1, c2, c3 = _busemann_coeffs(ma_inf)
    phi = float(phi_rad)
    return c1 * phi + c2 * phi**2 + c3 * phi**3

