"""Windward edge-cache for faceted 3D (sx, sy) slopes.

This is a minimal 3D upgrade that preserves the original solver architecture:
- keep the same reference-enthalpy "soup"
- replace the 2D windward slope dz/dx with a facet-normal based inflow angle phi(sx, sy)

Coordinate convention:
- x: streamwise (chordwise)
- y: spanwise
- z: upward

Definitions:
- surface slopes: sx = dz/dx, sy = dz/dy
- (unnormalized) facet normal used here: n = (sx, sy, 1)
- effective AoA with sweep: alpha_e (same as 2D solver, via independence principle)
- incoming unit flow direction (no sideslip): u = (cos(alpha_e), 0, -sin(alpha_e))

We choose phi such that it *exactly* reduces to the baseline 2D definition when sy=0:
    phi_2d = alpha_e - atan(sx)

Derivation (sy=0):
    s = - u·n_hat = (sin(alpha_e) - sx*cos(alpha_e)) / sqrt(1+sx^2) = sin(phi_2d)
    => phi = asin(s) = alpha_e - atan(sx)   (within principal range)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..aero.busemann import busemann_cp
from ..aero.edge_conditions import compute_edge_conditions, effective_alpha, effective_ma_inf
from ..aero.transition import transition_reynolds, transition_weight
from ..config.lf_qw import LfQwConfig
from ..heatflux.windward import windward_ref_enthalpy_branches
from ..types import EdgeConditions, GasModel


@dataclass(frozen=True)
class WindwardEdgeCacheFaceted3D:
    edges: list[EdgeConditions]  # length nx
    x_over_c: np.ndarray  # (nx,)
    x_phys: np.ndarray  # (nx,)
    lf_cfg: LfQwConfig
    transition_x_over_c: float | None


def _phi_from_slopes_3d(*, alpha_e: float, sx: float, sy: float) -> float:
    """3D inflow angle phi using (sx, sy) facet slopes.

    Returns phi in radians (principal asin range).
    """

    a = float(alpha_e)
    sx = float(sx)
    sy = float(sy)
    # n = (sx, sy, 1) and u = (cos a, 0, -sin a)
    # s = -u·n_hat
    denom = float(np.sqrt(1.0 + sx * sx + sy * sy))
    if not (denom > 0.0):
        denom = 1.0
    s = (float(np.sin(a)) - sx * float(np.cos(a))) / denom
    s = float(np.clip(s, -1.0, 1.0))
    return float(np.arcsin(s))


def build_windward_edge_cache_faceted3d(
    *,
    gas: GasModel,
    lf_cfg: LfQwConfig,
    mach: float,
    alpha_deg: float,
    sweep_le_deg: float,
    p_inf: float,
    rho_inf: float,
    T_inf: float,
    chord_m: float,
    xc_grid: np.ndarray,
    sx_arr: np.ndarray,
    sy_arr: np.ndarray,
    transition_x_over_c: float | None,
) -> WindwardEdgeCacheFaceted3D:
    """Compute edge conditions along one windward strip for faceted 3D geometry."""

    chi_rad = float(np.deg2rad(float(sweep_le_deg)))
    alpha_rad = float(np.deg2rad(float(alpha_deg)))
    alpha_e = float(effective_alpha(alpha_rad, chi_rad))
    mach_eff = float(effective_ma_inf(float(mach), alpha_rad=alpha_rad, chi_w_rad=chi_rad))

    x_over_c = np.asarray(xc_grid, dtype=float).reshape(-1)
    sx_arr = np.asarray(sx_arr, dtype=float).reshape(-1)
    sy_arr = np.asarray(sy_arr, dtype=float).reshape(-1)
    if sx_arr.size != x_over_c.size or sy_arr.size != x_over_c.size:
        raise ValueError("sx_arr and sy_arr must have the same length as xc_grid")

    # Internal x clamp for strip-theory evaluation (does not affect sampling grid)
    x_min_over_c = float(getattr(lf_cfg, "x_model").x_min_over_c) if hasattr(lf_cfg, "x_model") else 0.003
    x_eff_over_c = np.maximum(x_over_c, max(x_min_over_c, 0.0))
    x_phys = np.maximum(x_eff_over_c * float(chord_m), 1e-6)

    # phi clamp
    phi_clamp = bool(lf_cfg.phi_clamp.enable)
    phi_min = float(lf_cfg.phi_clamp.phi_min_rad)

    # Cp0 at leading edge should not depend on the first sampling point.
    # If x/c does not start at 0, extrapolate (sx, sy) to x/c=0 using the first two points.
    sx0 = float(sx_arr[0])
    sy0 = float(sy_arr[0])
    try:
        if x_over_c.size >= 2 and float(x_over_c[0]) > 0.0:
            x0 = float(x_over_c[0])
            x1 = float(x_over_c[1])
            if x1 != x0:
                sx0 = float(sx_arr[0] + (0.0 - x0) * (float(sx_arr[1]) - float(sx_arr[0])) / (x1 - x0))
                sy0 = float(sy_arr[0] + (0.0 - x0) * (float(sy_arr[1]) - float(sy_arr[0])) / (x1 - x0))
    except Exception:
        sx0 = float(sx_arr[0])
        sy0 = float(sy_arr[0])

    phi0 = _phi_from_slopes_3d(alpha_e=alpha_e, sx=sx0, sy=sy0)
    if phi_clamp and phi0 <= phi_min:
        phi0 = phi_min
    cp0 = busemann_cp(ma_inf=mach_eff, phi_rad=float(phi0))

    edges: list[EdgeConditions] = []
    for i in range(x_over_c.size):
        phi = _phi_from_slopes_3d(alpha_e=alpha_e, sx=float(sx_arr[i]), sy=float(sy_arr[i]))
        if phi_clamp and phi <= phi_min:
            phi = phi_min

        cp = busemann_cp(ma_inf=mach_eff, phi_rad=float(phi))
        edge = compute_edge_conditions(
            gas=gas,
            ma_inf=mach_eff,
            p_inf=float(p_inf),
            T_inf=float(T_inf),
            rho_inf=float(rho_inf),
            cp_pressure=float(cp),
            cp0_pressure=float(cp0),
        )
        edges.append(edge)

    return WindwardEdgeCacheFaceted3D(
        edges=edges,
        x_over_c=x_over_c,
        x_phys=x_phys,
        lf_cfg=lf_cfg,
        transition_x_over_c=transition_x_over_c,
    )


def windward_q_distribution_from_Tw(
    *,
    gas: GasModel,
    cache: WindwardEdgeCacheFaceted3D,
    Tw: np.ndarray,
    include_leading_edge: bool = False,
) -> np.ndarray:
    """Same semantics as aero/windward_cache.py (faceted 3D cache variant)."""

    Tw = np.asarray(Tw, dtype=float).reshape(-1)
    if Tw.size != cache.x_over_c.size:
        raise ValueError("Tw must have same length as cache.x_over_c")

    q = np.full_like(Tw, np.nan, dtype=float)
    i0 = 0 if bool(include_leading_edge) else 1
    for i in range(i0, Tw.size):
        Tw_i = float(Tw[i])
        if not np.isfinite(Tw_i):
            continue
        h_w = float(gas.h_from_T(Tw_i))
        q_lam, q_turb, re_lam, _re_turb = windward_ref_enthalpy_branches(
            gas=gas, edge=cache.edges[i], x=float(cache.x_phys[i]), h_w=h_w
        )
        re_tri = float(transition_reynolds(ma_e=float(cache.edges[i].ma_e)))
        w = float(
            transition_weight(
                enable=bool(cache.lf_cfg.transition.enable),
                re_measure=float(re_lam),
                re_tri=re_tri,
                weighting=str(cache.lf_cfg.transition.weighting),
                width_decades=float(cache.lf_cfg.transition.width_decades),
                x_over_c=float(cache.x_over_c[i]),
                transition_x_over_c=cache.transition_x_over_c,
            )
        )
        q[i] = (1.0 - w) * float(q_lam) + w * float(q_turb)
    return q


def windward_q_at_index(
    *,
    gas: GasModel,
    cache: WindwardEdgeCacheFaceted3D,
    i: int,
    Tw_i: float,
) -> float:
    """Same semantics as aero/windward_cache.py (faceted 3D cache variant)."""

    i = int(i)
    if i < 0:
        raise ValueError("windward_q_at_index expects i>=0.")
    if i >= cache.x_over_c.size:
        raise IndexError("i out of range for cache")
    Tw_i = float(Tw_i)
    if not np.isfinite(Tw_i):
        return float("nan")
    h_w = float(gas.h_from_T(Tw_i))
    q_lam, q_turb, re_lam, _re_turb = windward_ref_enthalpy_branches(
        gas=gas, edge=cache.edges[i], x=float(cache.x_phys[i]), h_w=h_w
    )
    re_tri = float(transition_reynolds(ma_e=float(cache.edges[i].ma_e)))
    w = float(
        transition_weight(
            enable=bool(cache.lf_cfg.transition.enable),
            re_measure=float(re_lam),
            re_tri=re_tri,
            weighting=str(cache.lf_cfg.transition.weighting),
            width_decades=float(cache.lf_cfg.transition.width_decades),
            x_over_c=float(cache.x_over_c[i]),
            transition_x_over_c=cache.transition_x_over_c,
        )
    )
    return float((1.0 - w) * float(q_lam) + w * float(q_turb))

