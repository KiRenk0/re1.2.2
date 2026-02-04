"""Config parsing for lf_qw_model section (keeps solver thin and behavior explicit)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..specs.models import CaseSpec


def _as_dict(x: Any) -> dict[str, Any]:
    return dict(x) if isinstance(x, dict) else {}


@dataclass(frozen=True)
class PhiClampConfig:
    enable: bool = True
    warn: bool = True
    phi_min_rad: float = 1e-8

    @classmethod
    def from_case(cls, case: CaseSpec) -> "PhiClampConfig":
        edge_model = _as_dict(case.lf_qw_model).get("edge_model", {})
        edge_model = _as_dict(edge_model)
        return cls(
            enable=bool(edge_model.get("phi_clamp", True)),
            warn=bool(edge_model.get("phi_warn", True)),
            phi_min_rad=float(edge_model.get("phi_min_rad", 1e-8)),
        )


@dataclass(frozen=True)
class TransitionBlendConfig:
    enable: bool = True
    weighting: str = "logistic"  # "logistic" or "step"
    width_decades: float = 0.25

    @classmethod
    def from_case(cls, case: CaseSpec) -> "TransitionBlendConfig":
        tr = _as_dict(case.lf_qw_model).get("transition", {})
        tr = _as_dict(tr)
        return cls(
            enable=bool(tr.get("enable", True)),
            weighting=str(tr.get("weighting", "logistic")).strip().lower(),
            width_decades=float(tr.get("width_decades", 0.25)),
        )


@dataclass(frozen=True)
class XModelConfig:
    """Engineering guards for x->0 behavior in strip-theory formulas."""

    # Minimum x/c used internally in formulas that behave singular as x->0
    # (e.g. Re_x ~ x and q ~ Re_x^-1/2). This does NOT change the sampling grid;
    # it only affects the model evaluation.
    x_min_over_c: float = 0.003

    @classmethod
    def from_case(cls, case: CaseSpec) -> "XModelConfig":
        edge_model = _as_dict(case.lf_qw_model).get("edge_model", {})
        edge_model = _as_dict(edge_model)
        x_min = float(edge_model.get("x_min_over_c", 0.003))
        # Keep it non-negative; 0 means "no clamp" (not recommended for sharp LE).
        if not (x_min >= 0.0):
            x_min = 0.003
        return cls(x_min_over_c=x_min)


@dataclass(frozen=True)
class StagnationConfig:
    rn_unit: str = "cm"  # "cm" or "m"
    sweep_exponent_n: float = 1.5

    @classmethod
    def from_case(cls, case: CaseSpec) -> "StagnationConfig":
        stag = _as_dict(case.lf_qw_model).get("stagnation", {})
        stag = _as_dict(stag)
        rn_unit = str(stag.get("rn_unit", "cm")).strip().lower()
        if rn_unit not in {"cm", "m"}:
            rn_unit = "cm"
        n = float(stag.get("sweep_exponent_n", 1.5))
        if not (n > 0):
            n = 1.5
        return cls(rn_unit=rn_unit, sweep_exponent_n=n)


@dataclass(frozen=True)
class LfQwConfig:
    phi_clamp: PhiClampConfig
    transition: TransitionBlendConfig
    x_model: XModelConfig
    stagnation: StagnationConfig
    q_stag_ratio_warn: float = 2.0

    @classmethod
    def from_case(cls, case: CaseSpec) -> "LfQwConfig":
        w = _as_dict(case.lf_qw_model).get("warnings", {})
        w = _as_dict(w)
        return cls(
            phi_clamp=PhiClampConfig.from_case(case),
            transition=TransitionBlendConfig.from_case(case),
            x_model=XModelConfig.from_case(case),
            stagnation=StagnationConfig.from_case(case),
            q_stag_ratio_warn=float(w.get("q_max_over_q_stag", 2.0)),
        )

