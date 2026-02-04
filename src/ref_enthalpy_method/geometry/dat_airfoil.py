"""Airfoil geometry loaded from .dat files (compat with ref_enthalpy baseline).

Baseline behavior:
- First line is a name/comment (skipped)
- Remaining lines are two floats: x y
- Upper surface: y >= 0, lower surface: y < 0
- Prefer SciPy CubicSpline; fall back to linear interpolation
- Compute slopes dy/dx on xc_grid and clip to avoid leading-edge numerical singularities
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..specs.loader import SpecError


@dataclass(frozen=True)
class AirfoilGeometry:
    """Airfoil representation on a given chordwise grid (xc_grid)."""

    spline_up: Callable[[np.ndarray], np.ndarray]
    spline_lo: Callable[[np.ndarray], np.ndarray]
    slope_up: np.ndarray  # dy/dx on xc_grid
    slope_lo: np.ndarray  # dy/dx on xc_grid
    t_over_c: float


def _try_cubic_spline():
    try:
        from scipy.interpolate import CubicSpline  # type: ignore

        return CubicSpline
    except Exception:
        return None


def _unique_sorted_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.argsort(x)
    x_s, y_s = x[idx], y[idx]
    x_u, idx_u = np.unique(x_s, return_index=True)
    y_u = y_s[idx_u]
    return x_u, y_u


def _linear_interp(x_u: np.ndarray, y_u: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    def f(xq):
        xq = np.asarray(xq, dtype=float)
        return np.interp(xq, x_u, y_u)

    return f


def _linear_slope(x_u: np.ndarray, y_u: np.ndarray, xc_grid: np.ndarray) -> np.ndarray:
    xc = np.asarray(xc_grid, dtype=float)
    x_u = np.asarray(x_u, dtype=float)
    y_u = np.asarray(y_u, dtype=float)
    j = np.searchsorted(x_u, xc, side="right")
    j = np.clip(j, 1, len(x_u) - 1)
    x0 = x_u[j - 1]
    x1 = x_u[j]
    y0 = y_u[j - 1]
    y1 = y_u[j]
    denom = np.where(np.abs(x1 - x0) < 1e-12, 1e-12, (x1 - x0))
    return (y1 - y0) / denom


def load_airfoil_dat_geometry(
    dat_path: str | Path,
    *,
    xc_grid: np.ndarray,
    slope_clip: float = 10.0,
) -> AirfoilGeometry:
    """Load an airfoil .dat and compute splines and slopes on xc_grid."""

    p = Path(dat_path)
    if not p.exists():
        raise SpecError(f"Airfoil .dat not found: {p}")

    # Optional header directives (baseline-compatible extension).
    # If the first line contains "interp=linear", force linear interpolation even if SciPy is available.
    # This is useful for sharp-corner airfoils (e.g., double-wedge/diamond) where cubic splines would
    # artificially round the kink and change slopes.
    header_line = ""
    try:
        with open(p, "r", encoding="utf-8") as f:
            header_line = (f.readline() or "").strip().lower()
    except Exception:
        header_line = ""

    try:
        data = np.loadtxt(p, skiprows=1)
    except Exception as e:
        raise SpecError(f"Failed to load airfoil .dat: {p} ({e})") from e

    if data.ndim != 2 or data.shape[1] < 2:
        raise SpecError(f"Invalid airfoil .dat format: {p}")

    x_all = np.asarray(data[:, 0], dtype=float)
    y_all = np.asarray(data[:, 1], dtype=float)
    mask_upper = y_all >= 0
    mask_lower = y_all < 0

    x_up, y_up = x_all[mask_upper], y_all[mask_upper]
    x_lo, y_lo = x_all[mask_lower], y_all[mask_lower]

    xc = np.asarray(xc_grid, dtype=float)
    CubicSpline = None if ("interp=linear" in header_line) else _try_cubic_spline()

    if CubicSpline is not None:
        x_u, y_u = _unique_sorted_xy(x_up, y_up)
        spline_up = CubicSpline(x_u, y_u)
        slope_up = spline_up(xc, 1)

        if len(x_lo) > 2:
            x_lu, y_lu = _unique_sorted_xy(x_lo, y_lo)
            spline_lo = CubicSpline(x_lu, y_lu)
            slope_lo = spline_lo(xc, 1)
        else:
            spline_lo = lambda x: -spline_up(x)
            slope_lo = -slope_up

        y_up_g = spline_up(xc)
        y_lo_g = spline_lo(xc)
    else:
        x_u, y_u = _unique_sorted_xy(x_up, y_up)
        spline_up = _linear_interp(x_u, y_u)
        slope_up = _linear_slope(x_u, y_u, xc)

        if len(x_lo) > 2:
            x_lu, y_lu = _unique_sorted_xy(x_lo, y_lo)
            spline_lo = _linear_interp(x_lu, y_lu)
            slope_lo = _linear_slope(x_lu, y_lu, xc)
        else:
            spline_lo = lambda x: -spline_up(x)
            slope_lo = -slope_up

        y_up_g = spline_up(xc)
        y_lo_g = spline_lo(xc)

    slope_up = np.clip(np.asarray(slope_up, dtype=float), -float(slope_clip), float(slope_clip))
    slope_lo = np.clip(np.asarray(slope_lo, dtype=float), -float(slope_clip), float(slope_clip))
    t_over_c = float(np.max(np.asarray(y_up_g, dtype=float) - np.asarray(y_lo_g, dtype=float)))

    return AirfoilGeometry(
        spline_up=spline_up,
        spline_lo=spline_lo,
        slope_up=slope_up,
        slope_lo=slope_lo,
        t_over_c=t_over_c,
    )

