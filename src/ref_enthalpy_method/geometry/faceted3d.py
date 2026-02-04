"""Simple faceted 3D lifting-body geometry helpers.

This module intentionally targets an *engineering* 3D upgrade path:
- Keep the original strip-theory + reference-enthalpy "soup"
- Replace the 2D slope (dz/dx) with a 3D facet normal derived from (dz/dx, dz/dy)
- Support a common "ridge-to-leading-edge" faceted assumption used when detailed CAD is unavailable.

Coordinate convention (body axes):
- x: streamwise, from nose to tail (m)
- y: spanwise, from centerline to one side (m, half-body)
- z: upward (m)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Faceted3DConfig:
    """Parameters for a symmetric ridge-to-edge faceted model."""

    ridge_up_deg: float
    ridge_lo_deg: float
    planform_half_angle_deg: float

    # Optional direct overrides for slopes (if provided, angles are ignored for that slope component).
    slope_x_up: float | None = None
    slope_y_up: float | None = None
    slope_x_lo: float | None = None
    slope_y_lo: float | None = None

    # Mask / numerical guards
    chord_min_m: float = 0.01

    # Optional: planform outline polyline (half-planform) for robust masking/chord extraction.
    # If provided, it overrides the triangle planform assumption.
    outline_csv_path: str | None = None
    outline_x_col: str = "x_m"
    outline_span_col: str = "y_m"  # can also be "z_m" depending on CAD export
    outline_span_sign: float = 1.0  # span_m = outline_span_sign * value_in_csv

    # Optional: surface mesh (ASCII STL) for sampling local facet slopes from CAD.
    surface_stl_path: str | None = None
    stl_unit: str = "auto"  # "auto" | "mm" | "m"
    stl_span_sign: float = -1.0  # solver_span_m = stl_span_sign * cad_z
    stl_right_half_only: bool = True

    @classmethod
    def from_faceted3d_spec(cls, d: dict) -> "Faceted3DConfig":
        """Parse a vehicle_spec.faceted3d mapping.

        Supported key aliases (to reduce spec ambiguity):
        - ridge_up_deg / ridge_lo_deg / planform_half_angle_deg (required)
        - slope overrides:
          - preferred: sx_up, sy_up, sx_lo, sy_lo
          - also accepted: slope_x_up, slope_y_up, slope_x_lo, slope_y_lo
        - chord_min_m (optional)
        """

        if not isinstance(d, dict):
            raise TypeError("faceted3d spec must be a mapping")

        def _get_float(key: str) -> float:
            if key not in d:
                raise KeyError(f"Missing faceted3d field: {key}")
            return float(d[key])

        def _opt_float(*keys: str) -> float | None:
            for k in keys:
                if k in d and d[k] is not None:
                    return float(d[k])
            return None

        planform = d.get("planform", {}) if isinstance(d.get("planform", {}), dict) else {}
        outline = planform.get("outline_csv", None) if isinstance(planform, dict) else None
        outline = None if outline in (None, "", False) else str(outline)
        outline_x_col = str(planform.get("outline_x_col", "x_m")) if isinstance(planform, dict) else "x_m"
        outline_span_col = str(planform.get("outline_span_col", "y_m")) if isinstance(planform, dict) else "y_m"
        outline_span_sign = float(planform.get("outline_span_sign", 1.0)) if isinstance(planform, dict) else 1.0

        surface = d.get("surface", {}) if isinstance(d.get("surface", {}), dict) else {}
        surface_stl = surface.get("stl", None) if isinstance(surface, dict) else None
        surface_stl = None if surface_stl in (None, "", False) else str(surface_stl)
        stl_unit = str(surface.get("unit", "auto")).strip().lower() if isinstance(surface, dict) else "auto"
        stl_span_sign = float(surface.get("span_sign", -1.0)) if isinstance(surface, dict) else -1.0
        stl_right_half_only = bool(surface.get("right_half_only", True)) if isinstance(surface, dict) else True

        return cls(
            ridge_up_deg=_get_float("ridge_up_deg"),
            ridge_lo_deg=_get_float("ridge_lo_deg"),
            planform_half_angle_deg=_get_float("planform_half_angle_deg"),
            slope_x_up=_opt_float("sx_up", "slope_x_up"),
            slope_y_up=_opt_float("sy_up", "slope_y_up"),
            slope_x_lo=_opt_float("sx_lo", "slope_x_lo"),
            slope_y_lo=_opt_float("sy_lo", "slope_y_lo"),
            chord_min_m=float(d.get("chord_min_m", 0.01)),
            outline_csv_path=outline,
            outline_x_col=outline_x_col,
            outline_span_col=outline_span_col,
            outline_span_sign=float(outline_span_sign),
            surface_stl_path=surface_stl,
            stl_unit=stl_unit,
            stl_span_sign=float(stl_span_sign),
            stl_right_half_only=bool(stl_right_half_only),
        )

    def slopes(self) -> tuple[float, float, float, float]:
        """Return (sx_up, sy_up, sx_lo, sy_lo).

        These are dimensionless slopes:
        - sx = dz/dx  (streamwise)
        - sy = dz/dy  (spanwise)

        Default faceted assumption (ridge-to-edge tent surface):
        - sx_up = tan(ridge_up_deg)
        - sy_up = -tan(ridge_up_deg)/tan(planform_half_angle_deg)
        - sx_lo = -tan(ridge_lo_deg)
        - sy_lo =  tan(ridge_lo_deg)/tan(planform_half_angle_deg)

        Any provided overrides (slope_x_*, slope_y_*) take precedence.
        """

        a_up = float(np.deg2rad(self.ridge_up_deg))
        a_lo = float(np.deg2rad(self.ridge_lo_deg))
        ha = float(np.deg2rad(self.planform_half_angle_deg))
        tan_ha = math.tan(ha) if abs(ha) > 1e-12 else 1e-12

        sx_up = math.tan(a_up)
        sy_up = -math.tan(a_up) / tan_ha
        sx_lo = -math.tan(a_lo)
        sy_lo = math.tan(a_lo) / tan_ha

        if self.slope_x_up is not None:
            sx_up = float(self.slope_x_up)
        if self.slope_y_up is not None:
            sy_up = float(self.slope_y_up)
        if self.slope_x_lo is not None:
            sx_lo = float(self.slope_x_lo)
        if self.slope_y_lo is not None:
            sy_lo = float(self.slope_y_lo)

        return float(sx_up), float(sy_up), float(sx_lo), float(sy_lo)


def triangle_x_le_from_half_angle(*, y_m: float, half_angle_deg: float) -> float:
    """Leading-edge x position for a triangular planform: x_le = y / tan(half_angle)."""

    y = float(y_m)
    ha = float(np.deg2rad(half_angle_deg))
    t = math.tan(ha)
    if abs(t) < 1e-12:
        return float("inf")
    return float(y / t)

    def slopes(self) -> tuple[float, float, float, float]:
        """Return (sx_up, sy_up, sx_lo, sy_lo)."""

        # Base from angles: tent surface between ridge line and planform edge line.
        a_up = float(np.deg2rad(self.ridge_up_deg))
        a_lo = float(np.deg2rad(self.ridge_lo_deg))
        ha = float(np.deg2rad(self.planform_half_angle_deg))
        tan_ha = math.tan(ha) if abs(ha) > 1e-12 else 1e-12

        sx_up = math.tan(a_up)
        sy_up = -math.tan(a_up) / tan_ha
        sx_lo = -math.tan(a_lo)
        sy_lo = math.tan(a_lo) / tan_ha

        # Overrides (if provided)
        if self.slope_x_up is not None:
            sx_up = float(self.slope_x_up)
        if self.slope_y_up is not None:
            sy_up = float(self.slope_y_up)
        if self.slope_x_lo is not None:
            sx_lo = float(self.slope_x_lo)
        if self.slope_y_lo is not None:
            sy_lo = float(self.slope_y_lo)
        return float(sx_up), float(sy_up), float(sx_lo), float(sy_lo)


def triangle_chord_from_half_angle(*, c_root_m: float, y_m: float, half_angle_deg: float) -> float:
    """Chord length at spanwise station y for a triangular planform.

    Model:
    - Leading edge line: x_le(y) = y / tan(half_angle)
    - Trailing edge is at x = c_root (fixed)
    => chord(y) = c_root - x_le(y)
    """

    c_root = float(c_root_m)
    y = float(y_m)
    ha = float(np.deg2rad(half_angle_deg))
    t = math.tan(ha)
    if abs(t) < 1e-12:
        return 0.0
    x_le = y / t
    return float(c_root - x_le)


def triangle_inside_mask(
    *,
    x_over_c: np.ndarray,
    y_over_b: float,
    c_root_m: float,
    b_half_m: float,
    half_angle_deg: float,
    chord_min_m: float,
) -> tuple[float, np.ndarray]:
    """Return (chord_m, mask_x) for one spanwise strip on a triangular planform."""

    y_m = float(y_over_b) * float(b_half_m)
    chord = triangle_chord_from_half_angle(c_root_m=float(c_root_m), y_m=y_m, half_angle_deg=float(half_angle_deg))
    chord = float(chord)
    if chord < float(chord_min_m):
        # Strip is essentially outside or degenerate near the tip.
        return float(chord), np.zeros((int(np.asarray(x_over_c).size),), dtype=bool)

    # For a valid strip, x/c in [0,1] is inside by construction.
    # We still allow caller to mask leading edge singular point if desired.
    return float(chord), np.ones((int(np.asarray(x_over_c).size),), dtype=bool)


def triangle_strip_xle_chord_mask(
    *,
    x_over_c: np.ndarray,
    y_over_b: float,
    c_root_m: float,
    b_half_m: float,
    half_angle_deg: float,
    chord_min_m: float,
) -> tuple[float, float, np.ndarray]:
    """Return (x_le_m, chord_m, mask_x) for one triangular-planform strip."""

    y_m = float(y_over_b) * float(b_half_m)
    x_le = triangle_x_le_from_half_angle(y_m=y_m, half_angle_deg=float(half_angle_deg))
    chord = triangle_chord_from_half_angle(c_root_m=float(c_root_m), y_m=y_m, half_angle_deg=float(half_angle_deg))
    chord = float(chord)
    if not np.isfinite(x_le) or chord < float(chord_min_m):
        return float(x_le), float(chord), np.zeros((int(np.asarray(x_over_c).size),), dtype=bool)
    return float(x_le), float(chord), np.ones((int(np.asarray(x_over_c).size),), dtype=bool)


def load_outline_csv(
    *,
    csv_path: str | Path,
    x_col: str = "x_m",
    span_col: str = "y_m",
    span_sign: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a planform outline polyline from CSV.

    The CSV must have a header row with at least (x_col, span_col). Units: meters.
    Returns (x_m, span_m) as 1D float arrays of equal length.
    """

    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"outline csv not found: {p}")

    # Use genfromtxt with names to avoid pandas dependency.
    data = np.genfromtxt(p, delimiter=",", names=True, dtype=float, encoding="utf-8")
    if data is None or getattr(data, "dtype", None) is None or data.dtype.names is None:
        raise ValueError(f"Failed to parse outline csv (need header): {p}")

    names = set([str(n) for n in data.dtype.names])
    if x_col not in names or span_col not in names:
        raise KeyError(f"outline csv missing columns. have={sorted(names)}, need=({x_col},{span_col})")

    x = np.asarray(data[x_col], dtype=float).reshape(-1)
    s = np.asarray(data[span_col], dtype=float).reshape(-1) * float(span_sign)

    # Drop non-finite rows
    ok = np.isfinite(x) & np.isfinite(s)
    x = x[ok]
    s = s[ok]
    if x.size < 3:
        raise ValueError(f"outline csv too short after filtering: {p}")

    return x, s


def _close_polyline(x: np.ndarray, y: np.ndarray, tol: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError("polyline x and y must have same length")
    if x.size < 3:
        raise ValueError("polyline too short")
    if abs(float(x[0]) - float(x[-1])) <= float(tol) and abs(float(y[0]) - float(y[-1])) <= float(tol):
        return x, y
    return np.concatenate([x, x[:1]]), np.concatenate([y, y[:1]])


def outline_strip_mask_and_chord(
    *,
    x_over_c: np.ndarray,
    y_over_b: float,
    b_half_m: float,
    outline_x_m: np.ndarray,
    outline_span_m: np.ndarray,
    chord_min_m: float,
) -> tuple[float, np.ndarray]:
    """Return (chord_m, mask_x) for one strip using an outline polygon.

    The outline is expected to be a *half-planform* closed polyline in (x, span) coordinates
    with span>=0. The strip is at physical span y = y_over_b * b_half_m.

    Implementation:
    - intersect horizontal line span=y with polygon edges -> x intersection set
    - take min/max intersections -> x_le/x_te
    - chord = x_te - x_le; x/c in [0,1] maps to x = x_le + (x/c)*chord
    """

    y_m = float(y_over_b) * float(b_half_m)
    if not np.isfinite(y_m):
        return float("nan"), np.zeros((int(np.asarray(x_over_c).size),), dtype=bool)

    px, py = _close_polyline(outline_x_m, outline_span_m)

    # Reject strips outside span range early.
    y_min = float(np.nanmin(py))
    y_max = float(np.nanmax(py))
    if not (y_min - 1e-9 <= y_m <= y_max + 1e-9):
        return 0.0, np.zeros((int(np.asarray(x_over_c).size),), dtype=bool)

    # Compute intersections with polygon edges.
    xs: list[float] = []
    for i in range(px.size - 1):
        x0, y0 = float(px[i]), float(py[i])
        x1, y1 = float(px[i + 1]), float(py[i + 1])

        # Skip horizontal edges to avoid infinite intersections.
        if abs(y1 - y0) < 1e-12:
            continue

        # Check if y_m is within [y0,y1) (half-open) to avoid double-counting vertices.
        if (y_m < min(y0, y1)) or (y_m >= max(y0, y1)):
            continue

        t = (y_m - y0) / (y1 - y0)
        if not (0.0 <= t <= 1.0):
            continue
        xs.append(x0 + t * (x1 - x0))

    if len(xs) < 2:
        return 0.0, np.zeros((int(np.asarray(x_over_c).size),), dtype=bool)

    xs_arr = np.asarray(xs, dtype=float)
    xs_arr = xs_arr[np.isfinite(xs_arr)]
    if xs_arr.size < 2:
        return 0.0, np.zeros((int(np.asarray(x_over_c).size),), dtype=bool)

    x_le = float(np.min(xs_arr))
    x_te = float(np.max(xs_arr))
    chord = float(x_te - x_le)
    if chord < float(chord_min_m):
        return float(chord), np.zeros((int(np.asarray(x_over_c).size),), dtype=bool)

    # All x/c points are valid if chord exists.
    return float(chord), np.ones((int(np.asarray(x_over_c).size),), dtype=bool)


def outline_strip_xle_chord_mask(
    *,
    x_over_c: np.ndarray,
    y_over_b: float,
    b_half_m: float,
    outline_x_m: np.ndarray,
    outline_span_m: np.ndarray,
    chord_min_m: float,
) -> tuple[float, float, np.ndarray]:
    """Return (x_le_m, chord_m, mask_x) for one strip using an outline polygon."""

    y_m = float(y_over_b) * float(b_half_m)
    if not np.isfinite(y_m):
        return float("nan"), float("nan"), np.zeros((int(np.asarray(x_over_c).size),), dtype=bool)

    px, py = _close_polyline(outline_x_m, outline_span_m)
    y_min = float(np.nanmin(py))
    y_max = float(np.nanmax(py))
    if not (y_min - 1e-9 <= y_m <= y_max + 1e-9):
        return 0.0, 0.0, np.zeros((int(np.asarray(x_over_c).size),), dtype=bool)

    xs: list[float] = []
    for i in range(px.size - 1):
        x0, y0 = float(px[i]), float(py[i])
        x1, y1 = float(px[i + 1]), float(py[i + 1])
        if abs(y1 - y0) < 1e-12:
            continue
        if (y_m < min(y0, y1)) or (y_m >= max(y0, y1)):
            continue
        t = (y_m - y0) / (y1 - y0)
        if not (0.0 <= t <= 1.0):
            continue
        xs.append(x0 + t * (x1 - x0))

    if len(xs) < 2:
        return 0.0, 0.0, np.zeros((int(np.asarray(x_over_c).size),), dtype=bool)

    xs_arr = np.asarray(xs, dtype=float)
    xs_arr = xs_arr[np.isfinite(xs_arr)]
    if xs_arr.size < 2:
        return 0.0, 0.0, np.zeros((int(np.asarray(x_over_c).size),), dtype=bool)

    x_le = float(np.min(xs_arr))
    x_te = float(np.max(xs_arr))
    chord = float(x_te - x_le)
    if chord < float(chord_min_m):
        return float(x_le), float(chord), np.zeros((int(np.asarray(x_over_c).size),), dtype=bool)

    return float(x_le), float(chord), np.ones((int(np.asarray(x_over_c).size),), dtype=bool)


def facet_normal_from_slopes(*, sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    """Return unit normals n_hat for z = sx*x + sy*y + const.

    n (unnormalized) = (-sx, -sy, 1)
    """

    sx = np.asarray(sx, dtype=float)
    sy = np.asarray(sy, dtype=float)
    nx = -sx
    ny = -sy
    nz = np.ones_like(nx, dtype=float)
    denom = np.sqrt(nx * nx + ny * ny + nz * nz)
    denom = np.where(denom <= 0.0, 1.0, denom)
    return np.stack([nx / denom, ny / denom, nz / denom], axis=-1)

