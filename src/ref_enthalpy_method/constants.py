"""Physical constants and default model parameters."""

from __future__ import annotations

SIGMA_BOLTZMANN = 57.6e-9  # W/(m^2 K^4), as used in the source doc (2.33)

# Sea-level reference values used in the modified Kemp-Riddell formula (2.40)
RHO_SL = 1.225  # kg/m^3
V_C = 7900.0  # m/s

# Default perfect-gas constants (can be overridden in configs)
GAMMA_AIR = 1.4
R_AIR = 287.0  # J/(kg K)

