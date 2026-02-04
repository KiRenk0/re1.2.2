"""Airfoil geometry helpers.

We need the local tangent angle to freestream:
    phi = alpha - arctan(f'(x))   (see eq. 2.13 in the doc)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def phi_from_slope(*, alpha_rad: float, dy_dx: float) -> float:
    """Compute phi (radians) from AoA and airfoil slope."""

    return float(alpha_rad) - math.atan(float(dy_dx))


@dataclass(frozen=True)
class DoubleWedgeAirfoil:
    """Very simple symmetric double-wedge (piecewise linear).

    This is a placeholder geometry for early verification runs.
    """

    chord: float
    t_over_c: float

    def y(self, x: float) -> float:
        # TODO: replace with the exact definition used in your reference case (Fig 2-13 etc.)
        x = float(x)
        c = float(self.chord)
        t = float(self.t_over_c) * c
        if x < 0 or x > c:
            return 0.0
        if x <= 0.5 * c:
            return (t / 2.0) * (x / (0.5 * c))
        return (t / 2.0) * (1.0 - (x - 0.5 * c) / (0.5 * c))

    def dy_dx(self, x: float) -> float:
        x = float(x)
        c = float(self.chord)
        t = float(self.t_over_c) * c
        if x < 0 or x > c:
            return 0.0
        if x < 0.5 * c:
            return (t / 2.0) / (0.5 * c)
        if x > 0.5 * c:
            return -(t / 2.0) / (0.5 * c)
        return 0.0


def make_double_wedge_airfoil(*, chord: float, t_over_c: float) -> DoubleWedgeAirfoil:
    return DoubleWedgeAirfoil(chord=chord, t_over_c=t_over_c)

