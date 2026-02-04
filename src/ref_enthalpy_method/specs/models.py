"""Spec dataclasses (minimal) aligned with ref_enthalpy/models/specs.py."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .loader import SpecError


def _req(d: dict[str, Any], path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise SpecError(f"Missing required field: {path}")
        cur = cur[part]
    return cur


def _opt(d: dict[str, Any], path: str, default: Any) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


@dataclass(frozen=True)
class VehicleSpec:
    vehicle_id: str
    b_half_m: float
    c_root_m: float
    c_tip_m: float
    sweep_le_deg: float
    rn_m: float
    airfoil_path: str
    emissivity: float = 0.8

    @classmethod
    def from_yaml_dict(cls, root: dict[str, Any]) -> "VehicleSpec":
        vs = _req(root, "vehicle_spec")
        if not isinstance(vs, dict):
            raise SpecError("vehicle_spec must be a mapping")
        return cls(
            vehicle_id=str(_req(vs, "vehicle_id")),
            b_half_m=float(_req(vs, "planform.b_half_m")),
            c_root_m=float(_req(vs, "planform.c_root_m")),
            c_tip_m=float(_req(vs, "planform.c_tip_m")),
            sweep_le_deg=float(_req(vs, "planform.sweep_le_deg")),
            rn_m=float(_req(vs, "leading_edge.rn_m")),
            airfoil_path=str(_req(vs, "airfoil.path")),
            emissivity=float(_opt(vs, "surface.emissivity", 0.8)),
        )


@dataclass(frozen=True)
class CaseSpec:
    gamma: float
    R_J_per_kgK: float
    pr: float
    cp_J_per_kgK: float
    fixed_h_m: float
    wall_temperature_K: float | None
    transition_x_over_c: float | None
    atmosphere_model: str
    tw_model_type: str
    sigma_W_m2_K4: float
    tw_transient: dict[str, Any]
    viscosity: dict[str, Any]
    lf_qw_model: dict[str, Any]

    @classmethod
    def from_yaml_dict(cls, root: dict[str, Any]) -> "CaseSpec":
        cs = _req(root, "case_spec")
        if not isinstance(cs, dict):
            raise SpecError("case_spec must be a mapping")
        lf = _req(cs, "lf_qw_model")
        if not isinstance(lf, dict):
            raise SpecError("case_spec.lf_qw_model must be a mapping")

        tw = _opt(cs, "tw_model", {}) or {}
        if not isinstance(tw, dict):
            raise SpecError("case_spec.tw_model must be a mapping if present")

        tw_transient = tw.get("transient", {}) if isinstance(tw, dict) else {}
        if tw_transient is None:
            tw_transient = {}
        if not isinstance(tw_transient, dict):
            raise SpecError("case_spec.tw_model.transient must be a mapping if present")

        wall = _opt(cs, "wall", {}) or {}
        wall_T = wall.get("temperature_K", None) if isinstance(wall, dict) else None
        wall_T = None if wall_T is None else float(wall_T)

        atm = _opt(cs, "atmosphere", {}) or {}
        if not isinstance(atm, dict):
            atm = {}
        atm_model = str(atm.get("model", "isa1976")).strip().lower()

        return cls(
            gamma=float(_req(cs, "gas.gamma")),
            R_J_per_kgK=float(_req(cs, "gas.R_J_per_kgK")),
            pr=float(_req(cs, "lf_qw_model.pr")),
            cp_J_per_kgK=float(_req(cs, "lf_qw_model.cp_J_per_kgK")),
            fixed_h_m=float(_req(cs, "fixed.h_m")),
            wall_temperature_K=wall_T,
            transition_x_over_c=(None if cs.get("transition_x_over_c", None) is None else float(cs["transition_x_over_c"])),
            atmosphere_model=atm_model,
            tw_model_type=str(tw.get("type", "")).strip().lower(),
            sigma_W_m2_K4=float(tw.get("sigma", 5.76e-8)),
            tw_transient=dict(tw_transient),
            viscosity=dict(_opt(cs, "viscosity", {}) or {}),
            lf_qw_model=dict(lf),
        )


@dataclass(frozen=True)
class SamplingSpec:
    mode: str
    x_start: float
    x_end: float
    nx: int
    y_start: float | None
    y_end: float | None
    ny: int
    concat_order: list[str]

    @classmethod
    def from_yaml_dict(cls, root: dict[str, Any]) -> "SamplingSpec":
        sp = _req(root, "canonical_sampling_spec")
        if not isinstance(sp, dict):
            raise SpecError("canonical_sampling_spec must be a mapping")

        mode = str(_req(sp, "mode"))
        x0 = float(_req(sp, "x_over_c.start"))
        x1 = float(_req(sp, "x_over_c.end"))
        nx = int(_req(sp, "x_over_c.n"))

        concat = sp.get("concat_order", sp.get("output_fields", ["q_w"]))
        concat_order = [str(x) for x in list(concat)]

        if mode == "root_windward_chord_line":
            return cls(mode=mode, x_start=x0, x_end=x1, nx=nx, y_start=None, y_end=None, ny=1, concat_order=concat_order)

        if mode == "full_wing_surface_grid":
            y0 = float(_req(sp, "y_over_b.start"))
            y1 = float(_req(sp, "y_over_b.end"))
            ny = int(_req(sp, "y_over_b.n"))
            return cls(mode=mode, x_start=x0, x_end=x1, nx=nx, y_start=y0, y_end=y1, ny=ny, concat_order=concat_order)

        # fallback
        y0 = sp.get("y_over_b", None)
        return cls(mode=mode, x_start=x0, x_end=x1, nx=nx, y_start=None, y_end=None, ny=1, concat_order=concat_order)

