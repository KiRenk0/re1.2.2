"""Shared datatypes (inputs/outputs) for the reference enthalpy method pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class FlightCondition:
    """Freestream condition."""

    altitude_m: float
    ma_inf: float
    alpha_rad: float = 0.0  # angle of attack
    chi_w_rad: float = 0.0  # leading-edge sweep


@dataclass(frozen=True)
class Material:
    """Skin material/structure properties used in (2.33)-(2.36) and balance (2.57)-(2.58)."""

    rho: float  # kg/m^3
    c: float  # J/(kg K)
    delta: float  # m
    emissivity: float  # epsilon


@dataclass(frozen=True)
class GasModel:
    """Gas model hooks.

    Minimal version: perfect gas with cp constant and Sutherland viscosity.
    """

    gamma: float
    R: float
    cp_gas: Callable[[float], float]  # cp(T) -> J/kg/K
    h_from_T: Callable[[float], float]  # h(T) -> J/kg
    T_from_h: Callable[[float], float]  # inverse mapping
    mu: Callable[[float], float]  # mu(T) -> Pa*s
    prandtl: float = 0.72


@dataclass(frozen=True)
class Airfoil:
    """2D airfoil shape definition for windward surface: y = f(x) and slope f'(x)."""

    y: Callable[[float], float]
    dy_dx: Callable[[float], float]


@dataclass(frozen=True)
class EdgeConditions:
    """Boundary-layer edge quantities at a surface point."""

    p_e: float
    rho_e: float
    T_e: float
    ma_e: float
    a_e: float
    v_e: float
    mu_e: float


@dataclass(frozen=True)
class HeatFluxResult:
    x: float
    q_a: float  # aerodynamic heating heat flux density
    regime: str  # "laminar" / "turbulent" / "log-law" etc.


@dataclass(frozen=True)
class WallTemperatureResult:
    x: float
    T_w: float
    q_a: float
    q_r: float


@dataclass(frozen=True)
class RunOptions:
    """Numerical options (iteration limits etc.)."""

    max_iter: int = 100
    tol: float = 1e-6
    # For unsteady (2.57) if used later
    dt: Optional[float] = None

