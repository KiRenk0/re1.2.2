"""Windward edge-cache + heat-flux evaluation helpers.

Goal: keep solver thin and make transient/steady windward computations reusable & testable.
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
class WindwardEdgeCache:
    edges: list[EdgeConditions]  # length nx
    x_over_c: np.ndarray  # (nx,)
    x_phys: np.ndarray  # (nx,)
    lf_cfg: LfQwConfig
    transition_x_over_c: float | None


def build_windward_edge_cache(
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
    slope_arr: np.ndarray,
    transition_x_over_c: float | None,
) -> WindwardEdgeCache:
    """Compute edge conditions along windward strip.

    Note: transition weighting depends on the Reynolds number used as a measure.
    In this project we take the *reference-enthalpy* Reynolds number from the laminar
    branch (Eq. 2.42) as the measure, which depends on wall enthalpy (thus Tw).
    Therefore the transition weight is computed on-the-fly when evaluating q(Tw),
    not during cache construction.
    """

    chi_rad = float(np.deg2rad(float(sweep_le_deg)))
    alpha_rad = float(np.deg2rad(float(alpha_deg)))
    alpha_e = effective_alpha(alpha_rad, chi_rad)
    mach_eff = effective_ma_inf(float(mach), alpha_rad=alpha_rad, chi_w_rad=chi_rad)

    x_over_c = np.asarray(xc_grid, dtype=float).reshape(-1)
    # Internal x clamp for strip-theory evaluation (does not affect sampling grid)
    x_min_over_c = float(getattr(lf_cfg, "x_model").x_min_over_c) if hasattr(lf_cfg, "x_model") else 0.003
    x_eff_over_c = np.maximum(x_over_c, max(x_min_over_c, 0.0))
    x_phys = np.maximum(x_eff_over_c * float(chord_m), 1e-6)

    # phi clamp
    phi_clamp = bool(lf_cfg.phi_clamp.enable)
    phi_min = float(lf_cfg.phi_clamp.phi_min_rad)

    # Cp0 at leading edge should not depend on the first sampling point.
    # If x/c does not start at 0, extrapolate slope to x/c=0 using the first two points.
    slope0 = float(slope_arr[0])
    try:
        if x_over_c.size >= 2 and float(x_over_c[0]) > 0.0:
            x0 = float(x_over_c[0])
            x1 = float(x_over_c[1])
            s0 = float(slope_arr[0])
            s1 = float(slope_arr[1])
            if x1 != x0:
                slope0 = s0 + (0.0 - x0) * (s1 - s0) / (x1 - x0)
    except Exception:
        slope0 = float(slope_arr[0])

    phi0 = float(alpha_e) - float(np.arctan(float(slope0)))
    if phi_clamp and phi0 <= phi_min:
        phi0 = phi_min
    cp0 = busemann_cp(ma_inf=mach_eff, phi_rad=phi0)

    edges: list[EdgeConditions] = []

    for i in range(x_over_c.size):
        phi = float(alpha_e) - float(np.arctan(float(slope_arr[i])))
        if phi_clamp and phi <= phi_min:
            phi = phi_min

        cp = busemann_cp(ma_inf=mach_eff, phi_rad=phi)
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

    return WindwardEdgeCache(
        edges=edges,
        x_over_c=x_over_c,
        x_phys=x_phys,
        lf_cfg=lf_cfg,
        transition_x_over_c=transition_x_over_c,
    )


def windward_q_distribution_from_Tw(
    *,
    gas: GasModel,
    cache: WindwardEdgeCache,
    Tw: np.ndarray,
    include_leading_edge: bool = False,
) -> np.ndarray:
    """Compute windward q(x) given Tw(x), using cached edges and transition weights.

    Note: leading-edge stagnation term is handled by the caller (solver), since it needs rho_inf/v_inf/h0 and rn_local.
    Here we return q for i>=1; q[0] is left as NaN for the caller to fill.
    """

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
        q_lam, q_turb, _re_lam, _re_turb = windward_ref_enthalpy_branches(
            gas=gas, edge=cache.edges[i], x=float(cache.x_phys[i]), h_w=h_w
        )
        re_tri = float(transition_reynolds(ma_e=float(cache.edges[i].ma_e)))
        w = float(
            transition_weight(
                enable=bool(cache.lf_cfg.transition.enable),
                re_measure=float(_re_lam),
                re_tri=re_tri,
                weighting=str(cache.lf_cfg.transition.weighting),
                width_decades=float(cache.lf_cfg.transition.width_decades),
                x_over_c=float(cache.x_over_c[i]),
                transition_x_over_c=cache.transition_x_over_c,
            )
        )
        q[i] = (1.0 - w) * q_lam + w * q_turb
    return q


def windward_q_at_index(
    *,
    gas: GasModel,
    cache: WindwardEdgeCache,
    i: int,
    Tw_i: float,
) -> float:
    """Compute windward q at a single index i>=1 for a given Tw_i.

    This is used to speed up steady radiative-equilibrium solving where Tw is solved pointwise.
    """

    i = int(i)
    if i < 0:
        raise ValueError("windward_q_at_index expects i>=0.")
    if i >= cache.x_over_c.size:
        raise IndexError("i out of range for cache")
    Tw_i = float(Tw_i)
    if not np.isfinite(Tw_i):
        return float("nan")
    h_w = float(gas.h_from_T(Tw_i))
    q_lam, q_turb, _re_lam, _re_turb = windward_ref_enthalpy_branches(
        gas=gas, edge=cache.edges[i], x=float(cache.x_phys[i]), h_w=h_w
    )
    re_tri = float(transition_reynolds(ma_e=float(cache.edges[i].ma_e)))
    w = float(
        transition_weight(
            enable=bool(cache.lf_cfg.transition.enable),
            re_measure=float(_re_lam),
            re_tri=re_tri,
            weighting=str(cache.lf_cfg.transition.weighting),
            width_decades=float(cache.lf_cfg.transition.width_decades),
            x_over_c=float(cache.x_over_c[i]),
            transition_x_over_c=cache.transition_x_over_c,
        )
    )
    return float((1.0 - w) * q_lam + w * q_turb)

