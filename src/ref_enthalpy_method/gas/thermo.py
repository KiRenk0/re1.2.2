"""Thermodynamics utilities.

Used by:
- Reference enthalpy method: Eckert reference enthalpy (2.38)
- Wall enthalpy model: h_w = cp_gas * T_w (see note near 2.57-2.58)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PerfectGasThermo:
    """Minimal perfect-gas thermo: constant cp and linear enthalpy h=cp*T."""

    cp_const: float  # J/(kg K)

    def cp(self, T: float) -> float:  # noqa: ARG002 - kept for future variable-cp extension
        return float(self.cp_const)

    def h_from_T(self, T: float) -> float:
        return float(self.cp_const) * float(T)

    def T_from_h(self, h: float) -> float:
        return float(h) / float(self.cp_const)


def make_perfect_gas_thermo(cp_const: float = 1004.5) -> PerfectGasThermo:
    """Factory for the simplest air thermo model."""

    return PerfectGasThermo(cp_const=cp_const)

