"""Windward surface heat flux density using strip theory (eq. 2.42).

This module will be the "main" one for reproducing q_a(x) on windward surface.

Equation (2.42) is piecewise in Re_x:
- Re_x <= 1e5: laminar (0.332 * ...)
- 1e5 < Re_x < 1e7: turbulent (0.0296 * ...)
- Re_x > 1e7: log-law fit (0.185 * ... (log Re_x)^-2.584)

It also uses the reference enthalpy method (Eckert, eq. 2.38) to obtain starred properties
rho*, mu* evaluated at h*.
"""

from __future__ import annotations

import math

from ..types import EdgeConditions, GasModel


def eckert_reference_enthalpy(*, h_e: float, h_w: float, h_r: float) -> float:
    """Eq. (2.38): h* = h_e + 0.5(h_w-h_e) + 0.22(h_r-h_e)."""

    he = float(h_e)
    return he + 0.5 * (float(h_w) - he) + 0.22 * (float(h_r) - he)


def reynolds_x(*, rho_e: float, v_e: float, x: float, mu_e: float) -> float:
    """Convenience Re_x = rho_e * v_e * x / mu_e."""

    return float(rho_e) * float(v_e) * float(x) / float(mu_e)


def recovery_temperature(*, T_e: float, Pr: float, ma_e: float, gamma: float, branch: str) -> float:
    """Engineering recovery temperature model used with reference enthalpy heating.

    Common choice:
    - laminar: r = sqrt(Pr)
    - turbulent: r = Pr^(1/3)
    T_r = T_e * (1 + r*(gamma-1)/2 * Ma_e^2)
    """

    Pr = float(Pr)
    if branch == "laminar":
        r = math.sqrt(Pr)
    else:
        r = Pr ** (1.0 / 3.0)
    return float(T_e) * (1.0 + float(r) * (float(gamma) - 1.0) / 2.0 * float(ma_e) ** 2)


def windward_ref_enthalpy_branches(
    *,
    gas: GasModel,
    edge: EdgeConditions,
    x: float,
    h_w: float,
    q_scale_lam: float = 1.0,
    q_scale_turb: float = 1.0,
) -> tuple[float, float, float, float]:
    """Return laminar/turbulent branch heat flux and their reference Reynolds numbers.

    Implements doc Eq (2.42) with Eckert reference enthalpy (2.38), using recovery
    temperature models for h_r (engineering closure).
    """

    Pr = float(gas.prandtl)
    g = float(gas.gamma)

    x = max(float(x), 1e-6)

    # edge properties
    p_e = float(edge.p_e)
    rho_e = float(edge.rho_e)
    T_e = float(edge.T_e)
    v_e = float(edge.v_e)
    ma_e = float(edge.ma_e)
    mu_e = float(edge.mu_e)

    h_e = float(gas.h_from_T(T_e))

    # --- Laminar branch ---
    T_r_lam = recovery_temperature(T_e=T_e, Pr=Pr, ma_e=ma_e, gamma=g, branch="laminar")
    h_r_lam = float(gas.h_from_T(T_r_lam))
    h_star_lam = eckert_reference_enthalpy(h_e=h_e, h_w=h_w, h_r=h_r_lam)
    T_star_lam = float(gas.T_from_h(h_star_lam))
    rho_star_lam = p_e / (float(gas.R) * T_star_lam)
    mu_star_lam = float(gas.mu(T_star_lam))
    Re_x_star_lam = max(rho_star_lam * v_e * x / mu_star_lam, 1.0)
    q_lam = (
        0.332
        * (Pr ** (-2.0 / 3.0))
        * rho_e
        * v_e
        * (Re_x_star_lam ** (-0.5))
        * (h_r_lam - h_w)
        * math.sqrt((rho_star_lam * mu_star_lam) / (rho_e * mu_e))
    ) * float(q_scale_lam)

    # --- Turbulent branch ---
    T_r_turb = recovery_temperature(T_e=T_e, Pr=Pr, ma_e=ma_e, gamma=g, branch="turbulent")
    h_r_turb = float(gas.h_from_T(T_r_turb))
    h_star_turb = eckert_reference_enthalpy(h_e=h_e, h_w=h_w, h_r=h_r_turb)
    T_star_turb = float(gas.T_from_h(h_star_turb))
    rho_star_turb = p_e / (float(gas.R) * T_star_turb)
    mu_star_turb = float(gas.mu(T_star_turb))
    Re_x_star_turb = max(rho_star_turb * v_e * x / mu_star_turb, 1.0)

    if Re_x_star_turb > 1.0e7:
        term = 0.185 * (Pr ** (-2.0 / 3.0)) * rho_e * v_e * (math.log10(Re_x_star_turb) ** (-2.584)) * (
            h_r_turb - h_w
        )
    else:
        term = 0.0296 * (Pr ** (-2.0 / 3.0)) * rho_e * v_e * (Re_x_star_turb ** (-0.2)) * (h_r_turb - h_w)
    q_turb = term * ((rho_star_turb / rho_e) ** 0.8) * ((mu_star_turb / mu_e) ** 0.2) * float(q_scale_turb)

    return float(q_lam), float(q_turb), float(Re_x_star_lam), float(Re_x_star_turb)


def windward_heat_flux(
    *,
    gas: GasModel,
    edge: EdgeConditions,
    x: float,
    h_r: float,
    h_w: float,
    h_e: float,
) -> tuple[float, str]:
    """Compute q_a at windward surface location x using (2.42).

    Returns (q_a, regime_label).
    """

    Pr = float(gas.prandtl)
    rho_e = float(edge.rho_e)
    v_e = float(edge.v_e)
    mu_e = float(edge.mu_e)

    # reference enthalpy (2.38) -> starred temperature -> starred props
    h_star = eckert_reference_enthalpy(h_e=h_e, h_w=h_w, h_r=h_r)
    T_star = float(gas.T_from_h(h_star))
    rho_star = float(edge.p_e) / (float(gas.R) * T_star)  # perfect-gas closure at edge pressure
    mu_star = float(gas.mu(T_star))

    Re_x = reynolds_x(rho_e=rho_e, v_e=v_e, x=x, mu_e=mu_e)
    dh = float(h_r) - float(h_w)

    if Re_x <= 1.0e5:
        q = (
            0.332
            * (Pr ** (-2.0 / 3.0))
            * rho_e
            * v_e
            * (Re_x ** -0.5)
            * dh
            * math.sqrt((rho_star * mu_star) / (rho_e * mu_e))
        )
        return q, "laminar"

    if Re_x < 1.0e7:
        q = (
            0.0296
            * (Pr ** (-2.0 / 3.0))
            * rho_e
            * v_e
            * (Re_x ** -0.2)
            * dh
            * ((rho_star / rho_e) ** 0.8)
            * ((mu_star / mu_e) ** 0.2)
        )
        return q, "turbulent"

    q = (
        0.185
        * (Pr ** (-2.0 / 3.0))
        * rho_e
        * v_e
        * ((math.log10(Re_x)) ** -2.584)
        * dh
        * ((rho_star / rho_e) ** 0.8)
        * ((mu_star / mu_e) ** 0.2)
    )
    return q, "log-law"

