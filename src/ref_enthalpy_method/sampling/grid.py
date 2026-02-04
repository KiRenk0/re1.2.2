"""Sampling grid generation from SamplingSpec (compat with baseline specs)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..specs.models import SamplingSpec


@dataclass(frozen=True)
class SamplingGrids:
    xc_grid: np.ndarray  # (nx,)
    yb_grid: np.ndarray  # (ny,) normalized by half-span (0=root, 1=tip); ny=1 for 1D


def make_sampling_grids(spec: SamplingSpec) -> SamplingGrids:
    """Build xc_grid and yb_grid for supported sampling modes.

    Baseline modes:
    - root_windward_chord_line: yb_grid=[0.0], xc_grid linspace
    - full_wing_surface_grid: yb_grid linspace, xc_grid linspace
    """

    xc = np.linspace(float(spec.x_start), float(spec.x_end), int(spec.nx))
    if str(spec.mode) == "root_windward_chord_line":
        yb = np.array([0.0], dtype=float)
        return SamplingGrids(xc_grid=xc, yb_grid=yb)

    if str(spec.mode) == "full_wing_surface_grid":
        y0 = 0.0 if spec.y_start is None else float(spec.y_start)
        y1 = 1.0 if spec.y_end is None else float(spec.y_end)
        yb = np.linspace(y0, y1, int(spec.ny))
        return SamplingGrids(xc_grid=xc, yb_grid=yb)

    # fallback: treat as 1D
    return SamplingGrids(xc_grid=xc, yb_grid=np.array([0.0], dtype=float))

