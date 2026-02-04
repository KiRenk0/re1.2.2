"""Spec-driven low-fidelity wing aerothermal solver (rewritten, baseline-compatible).

This is the new "front door" that will eventually replace ref_enthalpy/solver.py,
while keeping the same workflow (specs -> run -> runs/<dir>/summary.json + fields.npz).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .aero.windward_cache import build_windward_edge_cache, windward_q_at_index, windward_q_distribution_from_Tw
from .aero.transition import transition_reynolds, transition_weight
from .gas.thermo import make_perfect_gas_thermo
from .gas.transport import mu_sutherland
from .geometry.dat_airfoil import load_airfoil_dat_geometry
from .heatflux.leading_edge import leading_edge_heat_flux_baseline
from .heatflux.windward import windward_ref_enthalpy_branches
from .heatflux.leeward import (
    leeward_heat_flux_distribution,
    leeward_re_ns,
    leeward_stanton_distribution,
    normal_shock_temperature_ratio,
)
from .sampling.grid import make_sampling_grids
from .specs.loader import load_yaml
from .specs.models import CaseSpec, SamplingSpec, VehicleSpec
from .thermal.transient import march_explicit_balance, march_explicit_balance_final, require_transient_material
from .thermal.windward_equilibrium import solve_windward_radiative_equilibrium
from .thermal.leeward_equilibrium import solve_leeward_radiative_equilibrium_coupled
from .types import GasModel
from .utils.warnings import WarningLog
from .config.lf_qw import LfQwConfig


@dataclass(frozen=True)
class SolverConfig:
    vehicle_config: str
    case_config: str
    sampling_config: str
    run_dir: str


class WingLowFidelitySolver:
    """Baseline-compatible solver façade.

    Design intent:
    - Keep this file thin over time by pushing physics into modules.
    - Maintain ref_enthalpy-like ergonomics for running specs.
    """

    def __init__(self, *, vehicle_config: str, case_config: str, sampling_config: str, run_dir: str):
        self.project_root = self._resolve_project_root()
        self.veh_path = (self.project_root / vehicle_config).resolve()
        self.case_path = (self.project_root / case_config).resolve()
        self.samp_path = (self.project_root / sampling_config).resolve()
        self.run_dir = (self.project_root / run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Warnings file (baseline-style)
        self.warning_log = WarningLog(path=self.run_dir / "lf_warnings.log", enabled=True)
        self.warning_log.reset_file()

        # Loaded specs
        self.veh_spec_raw = load_yaml(self.veh_path)
        self.case_spec_raw = load_yaml(self.case_path)
        self.sampling_spec_raw = load_yaml(self.samp_path)

        self.vehicle = VehicleSpec.from_yaml_dict(self.veh_spec_raw)
        self.case = CaseSpec.from_yaml_dict(self.case_spec_raw)
        self.sampling = SamplingSpec.from_yaml_dict(self.sampling_spec_raw)
        self.lf_cfg = LfQwConfig.from_case(self.case)

        # Gas model (currently perfect gas + constant cp + Sutherland)
        thermo = make_perfect_gas_thermo(cp_const=self.case.cp_J_per_kgK)
        self.gas = GasModel(
            gamma=self.case.gamma,
            R=self.case.R_J_per_kgK,
            cp_gas=thermo.cp,
            h_from_T=thermo.h_from_T,
            T_from_h=thermo.T_from_h,
            mu=mu_sutherland,
            prandtl=self.case.pr,
        )

        # Sampling grids
        grids = make_sampling_grids(self.sampling)
        self.xc_grid = grids.xc_grid
        self.yb_grid = grids.yb_grid
        self.nx = int(self.xc_grid.size)
        self.ny = int(self.yb_grid.size)

        # Airfoil geometry (slope arrays on xc_grid)
        airfoil_dat_path = (self.veh_path.parent / self.vehicle.airfoil_path).resolve()
        self.airfoil_geom = load_airfoil_dat_geometry(airfoil_dat_path, xc_grid=self.xc_grid)

        # Public result cache (baseline-style)
        self.last_fields: dict[str, Any] = {}

    def _warn(self, msg: str) -> None:
        self.warning_log.warn(msg)

    @staticmethod
    def _resolve_project_root() -> Path:
        """Heuristic: choose workspace root (the directory containing this file's parents)."""

        here = Path(__file__).resolve()
        # Prefer repo root (two levels up from src/ref_enthalpy_method/)
        # <root>/src/ref_enthalpy_method/solver.py -> parents[2] = <root>
        if len(here.parents) >= 3:
            return here.parents[2]
        return here.parent

    def _chord_at_y(self, y_over_b: float) -> float:
        # linear taper along half-span
        return float(self.vehicle.c_root_m) + (float(self.vehicle.c_tip_m) - float(self.vehicle.c_root_m)) * float(y_over_b)

    def _windward_slope(self, alpha_deg: float) -> np.ndarray:
        # Baseline convention: positive alpha => windward is lower surface
        return self.airfoil_geom.slope_lo if float(alpha_deg) >= 0.0 else self.airfoil_geom.slope_up

    def _leeward_slope(self, alpha_deg: float) -> np.ndarray:
        return self.airfoil_geom.slope_up if float(alpha_deg) >= 0.0 else self.airfoil_geom.slope_lo

    def _freestream(self, mach: float) -> tuple[float, float, float, float]:
        """Return (p_inf, rho_inf, T_inf, v_inf).

        For now we follow baseline case specs which provide h_m and atmosphere model.
        We use our ISA1976 implementation (supports wider range) but can switch per spec later.
        """

        model = str(self.case.atmosphere_model).strip().lower()
        if model == "ussa1976":
            from .atmosphere.ussa1976 import ussa1976_0_32km

            p_inf, rho_inf, T_inf = ussa1976_0_32km(h_m=self.case.fixed_h_m, R_gas_J_per_kgK=self.case.R_J_per_kgK)
        else:
            from .atmosphere.isa1976 import isa1976

            atm = isa1976(self.case.fixed_h_m, R=self.case.R_J_per_kgK)
            T_inf, p_inf, rho_inf = atm.T, atm.p, atm.rho
        v_inf = float(mach) * float(np.sqrt(float(self.case.gamma) * float(self.case.R_J_per_kgK) * float(T_inf)))
        return float(p_inf), float(rho_inf), float(T_inf), float(v_inf)

    def _leading_edge_q(
        self,
        *,
        rho_inf: float,
        v_inf: float,
        h0: float,
        h_w: float,
        alpha_rad: float,
        chi_rad: float,
        rn_local_m: float,
    ) -> float:
        """Leading-edge heat flux (baseline-compatible implementation).

        Notes:
        - Baseline supports rn_unit ('m' or 'cm') because the Kemp-Riddell constant is often used
          in W/cm^2 with RN in cm, then converted to W/m^2.
        - Baseline also scales RN with local chord: rn_local = rn_le * (chord / c_root).
          Here chord scaling is applied by the caller (passes already-scaled RN via self._rn_local()).
        """

        # Convert rn_local back into a chord length to reuse a single baseline helper
        rn_le = float(self.vehicle.rn_m)
        c_root = float(self.vehicle.c_root_m)
        chord_m = c_root if rn_le <= 0 else (float(rn_local_m) * c_root / rn_le)

        h_300K = float(self.gas.h_from_T(300.0))
        return leading_edge_heat_flux_baseline(
            rn_le_m=rn_le,
            c_root_m=c_root,
            chord_m=chord_m,
            rn_unit=str(self.lf_cfg.stagnation.rn_unit),
            sweep_exponent_n=float(self.lf_cfg.stagnation.sweep_exponent_n),
            rho_inf=float(rho_inf),
            v_inf=float(v_inf),
            h0=float(h0),
            h_w=float(h_w),
            h_300K=float(h_300K),
            chi_w_rad=float(chi_rad),
            alpha_rad=float(alpha_rad),
        )

    def _rn_local(self, *, chord_m: float) -> float:
        """Baseline chord scaling for leading-edge radius."""

        return float(self.vehicle.rn_m) * float(chord_m) / float(self.vehicle.c_root_m)

    def _leeward_R_ref(self, *, chord_m: float) -> float:
        """Reference length for leeward Re_ns correlation.

        Historically the baseline used leading-edge radius rn_m as R_ref. When we model a sharp leading edge
        (rn_m <= 0), Re_ns must still use a positive reference length. We fall back to the local chord.
        """

        rn = float(self.vehicle.rn_m)
        if rn > 0.0:
            return rn
        return max(float(chord_m), 1e-6)

    def calc_strip_heat_flux_fixed_wall(
        self,
        *,
        mach: float,
        alpha_deg: float,
        chord_m: float,
        side: str,
        T_wall_K: float,
    ) -> np.ndarray:
        """Compute q(x) for one strip with fixed wall temperature."""

        if side not in {"windward", "leeward"}:
            raise ValueError(f"Invalid side={side!r}")

        p_inf, rho_inf, T_inf, v_inf = self._freestream(mach)
        h_inf = float(self.gas.h_from_T(T_inf))
        h0 = h_inf + 0.5 * (v_inf**2)
        h_w = float(self.gas.h_from_T(float(T_wall_K)))

        # Fast-path: leeward is a mean correlation and doesn't depend on alpha/geometry here.
        if side == "leeward":
            h_s = float(h0)
            ratio_T = normal_shock_temperature_ratio(gamma=float(self.case.gamma), mach=float(mach))
            T_ns = float(T_inf) * float(ratio_T)
            mu_ns = float(self.gas.mu(T_ns))
            Re_ns = leeward_re_ns(rho_inf=rho_inf, v_inf=v_inf, R_ref=self._leeward_R_ref(chord_m=chord_m), mu_ns=mu_ns)
            h_wwd_dist = np.full((self.nx,), h_w, dtype=float)
            St_dist = leeward_stanton_distribution(Re_ns=float(Re_ns), h_wwd_dist=h_wwd_dist, h_s=float(h_s))
            q_dist = leeward_heat_flux_distribution(rho_inf=rho_inf, v_inf=v_inf, St_dist=St_dist, h_s=float(h_s), h_w=h_w)
            if not np.all(np.isfinite(q_dist)):
                bad = np.where(~np.isfinite(q_dist))[0]
                self._warn(f"NaN/Inf in leeward heat flux | M={mach:.2f}, indices={bad.tolist()}")
            return q_dist

        # Windward (uses edge cache)
        slope_arr = self._windward_slope(alpha_deg)

        # Precompute phi clamp warning preview (edge cache does the actual clamping internally too)
        phi_clamp_enable = bool(self.lf_cfg.phi_clamp.enable)
        phi_warn = bool(self.lf_cfg.phi_clamp.warn)
        phi_min = float(self.lf_cfg.phi_clamp.phi_min_rad)
        chi_rad = float(np.deg2rad(self.vehicle.sweep_le_deg))
        alpha_rad = float(np.deg2rad(alpha_deg))
        alpha_e = effective_alpha(alpha_rad, chi_rad)
        phi_arr = alpha_e - np.arctan(np.asarray(slope_arr, dtype=float).reshape(-1))
        clamped_idx = np.where(phi_clamp_enable & (phi_arr <= phi_min))[0]
        if phi_warn and clamped_idx.size > 0:
            preview = ", ".join([f"{float(self.xc_grid[i]):.4f}" for i in clamped_idx[:6].tolist()])
            self._warn(
                f"phi clamped at {int(clamped_idx.size)} points | M={mach:.2f}, alpha={alpha_deg:.2f}, side=windward | first x/c: [{preview}]"
            )

        cache = build_windward_edge_cache(
            gas=self.gas,
            lf_cfg=self.lf_cfg,
            mach=float(mach),
            alpha_deg=float(alpha_deg),
            sweep_le_deg=float(self.vehicle.sweep_le_deg),
            p_inf=float(p_inf),
            rho_inf=float(rho_inf),
            T_inf=float(T_inf),
            chord_m=float(chord_m),
            xc_grid=np.asarray(self.xc_grid, dtype=float),
            slope_arr=np.asarray(slope_arr, dtype=float),
            transition_x_over_c=self.case.transition_x_over_c,
        )

        Tw = np.full((self.nx,), float(T_wall_K), dtype=float)
        q_dist = windward_q_distribution_from_Tw(gas=self.gas, cache=cache, Tw=Tw)
        # Leading edge (index 0):
        # - If rn_m > 0: use blunt stagnation correlation (doc 2.39-2.41 / 2.40)
        # - If rn_m <= 0: treat as sharp leading edge and use strip-model proxy at x≈0
        if float(self.vehicle.rn_m) > 0.0:
            rn_local_m = self._rn_local(chord_m=chord_m)
            q_dist[0] = self._leading_edge_q(
                rho_inf=rho_inf,
                v_inf=v_inf,
                h0=h0,
                h_w=h_w,
                alpha_rad=alpha_rad,
                chi_rad=chi_rad,
                rn_local_m=rn_local_m,
            )
        else:
            q_dist[0] = windward_q_at_index(gas=self.gas, cache=cache, i=0, Tw_i=float(T_wall_K))

        if not np.all(np.isfinite(q_dist)):
            bad = np.where(~np.isfinite(q_dist))[0]
            self._warn(
                f"NaN/Inf in heat flux | M={mach:.2f}, alpha={alpha_deg:.2f}, side=windward | indices={bad.tolist()}"
            )
        if side == "windward" and np.isfinite(q_dist[0]) and np.any(np.isfinite(q_dist[1:])):
            q_stag = float(q_dist[0])
            q_max = float(np.nanmax(q_dist))
            ratio = float(self.lf_cfg.q_stag_ratio_warn)
            if q_stag > 0 and q_max > ratio * q_stag:
                self._warn(
                    f"High local heat flux (>{ratio:g}*q_stag): max/stag={q_max/q_stag:.2f} | M={mach:.2f}, alpha={alpha_deg:.2f}"
                )

        return q_dist

    # Windward edge cache is now in aero/windward_cache.py

    def calc_strip_radiative_equilibrium(self, *, mach: float, alpha_deg: float, chord_m: float, side: str) -> tuple[np.ndarray, np.ndarray]:
        """Compute (Tw(x), q(x)) using steady radiative equilibrium (2.58)."""

        if side != "windward":
            raise ValueError("This solver uses coupled leeward equilibrium; use windward here.")

        # Windward: build cache once, then solve Tw(i) pointwise using q(Tw) at that index.
        p_inf, rho_inf, T_inf, v_inf = self._freestream(mach)
        h_inf = float(self.gas.h_from_T(T_inf))
        h0 = h_inf + 0.5 * (float(v_inf) ** 2)
        chi_rad = float(np.deg2rad(self.vehicle.sweep_le_deg))
        alpha_rad = float(np.deg2rad(alpha_deg))

        slope_arr = self._windward_slope(alpha_deg)
        cache = build_windward_edge_cache(
            gas=self.gas,
            lf_cfg=self.lf_cfg,
            mach=float(mach),
            alpha_deg=float(alpha_deg),
            sweep_le_deg=float(self.vehicle.sweep_le_deg),
            p_inf=float(p_inf),
            rho_inf=float(rho_inf),
            T_inf=float(T_inf),
            chord_m=float(chord_m),
            xc_grid=np.asarray(self.xc_grid, dtype=float),
            slope_arr=np.asarray(slope_arr, dtype=float),
            transition_x_over_c=self.case.transition_x_over_c,
        )
        rn_local_m = self._rn_local(chord_m=chord_m)

        def q_leading_edge_of_Tw(Tw0: float) -> float:
            # See calc_strip_heat_flux_fixed_wall for sharp-vs-blunt leading-edge behavior.
            if float(self.vehicle.rn_m) > 0.0:
                h_w0 = float(self.gas.h_from_T(float(Tw0)))
                return self._leading_edge_q(
                    rho_inf=float(rho_inf),
                    v_inf=float(v_inf),
                    h0=float(h0),
                    h_w=float(h_w0),
                    alpha_rad=float(alpha_rad),
                    chi_rad=float(chi_rad),
                    rn_local_m=float(rn_local_m),
                )
            return windward_q_at_index(gas=self.gas, cache=cache, i=0, Tw_i=float(Tw0))

        return solve_windward_radiative_equilibrium(
            gas=self.gas,
            cache=cache,
            emissivity=float(self.vehicle.emissivity),
            sigma_W_m2_K4=float(self.case.sigma_W_m2_K4),
            q_leading_edge_of_Tw=q_leading_edge_of_Tw,
        )

    def compute_snapshot(self, *, mach: float, alpha: float) -> np.ndarray:
        """Compute outputs according to sampling spec concat_order (baseline-compatible)."""

        self.last_fields = {}
        tw_type = str(self.case.tw_model_type or "").strip().lower()

        # 1D or 2D grids: we return flattened arrays in y_then_x order for 2D
        q_w_list = []
        q_l_list = []
        Tw_w_list = []
        Tw_l_list = []
        w_tr_list = []
        re_edge_list = []
        re_tri_list = []

        for j in range(self.ny):
            yb = float(self.yb_grid[j])
            chord = self._chord_at_y(yb)

            if tw_type == "transient_balance":
                # Baseline: transient is typically applied to windward only, and returns time history.
                cfg = dict(self.case.tw_transient or {})
                require_transient_material(cfg)

                rho_wall = float(cfg["rho_wall_kg_m3"])
                c_wall = float(cfg["c_wall_J_per_kgK"])
                delta_wall = float(cfg["delta_wall_m"])
                cap = float(rho_wall * c_wall * delta_wall)

                dt = float(cfg.get("dt_s", 0.01))
                t_end = float(cfg.get("t_end_s", 10.0))
                n_steps = int(np.ceil(t_end / dt)) if t_end > 0 else 0
                t_s = np.linspace(0.0, n_steps * dt, n_steps + 1)

                Tw_init = cfg.get("Tw_init_K", self.case.wall_temperature_K if self.case.wall_temperature_K is not None else 300.0)
                if np.isscalar(Tw_init):
                    Tw0 = np.full((self.nx,), float(Tw_init), dtype=float)
                else:
                    Tw0 = np.asarray(Tw_init, dtype=float).reshape(-1)
                    if Tw0.size != self.nx:
                        raise ValueError(f"Tw_init_K must be scalar or length nx={self.nx}")

                Tw_min = float(cfg.get("Tw_min_K", 150.0))
                Tw_max = float(cfg.get("Tw_max_K", 6000.0))
                eps = float(self.vehicle.emissivity)
                sigma = float(self.case.sigma_W_m2_K4)

                # Precompute windward edge cache for speed; q_a depends on Tw only via h_w
                slope_arr = self._windward_slope(alpha)
                p_inf, rho_inf, T_inf, v_inf = self._freestream(mach)
                cache = build_windward_edge_cache(
                    gas=self.gas,
                    lf_cfg=self.lf_cfg,
                    mach=float(mach),
                    alpha_deg=float(alpha),
                    sweep_le_deg=float(self.vehicle.sweep_le_deg),
                    p_inf=float(p_inf),
                    rho_inf=float(rho_inf),
                    T_inf=float(T_inf),
                    chord_m=float(chord),
                    xc_grid=np.asarray(self.xc_grid, dtype=float),
                    slope_arr=np.asarray(slope_arr, dtype=float),
                    transition_x_over_c=self.case.transition_x_over_c,
                )
                rn_local_m = self._rn_local(chord_m=chord)
                h_inf = float(self.gas.h_from_T(T_inf))
                h0 = h_inf + 0.5 * (float(v_inf) ** 2)
                chi_rad = float(np.deg2rad(self.vehicle.sweep_le_deg))
                alpha_rad = float(np.deg2rad(alpha))

                def eval_q_a(Tw_k: np.ndarray) -> np.ndarray:
                    Tw_k = np.asarray(Tw_k, dtype=float).reshape(-1)
                    qk = windward_q_distribution_from_Tw(gas=self.gas, cache=cache, Tw=Tw_k)
                    # leading edge (index 0)
                    Tw0 = float(Tw_k[0]) if Tw_k.size > 0 else float("nan")
                    if np.isfinite(Tw0):
                        if float(self.vehicle.rn_m) > 0.0:
                            h_w0 = float(self.gas.h_from_T(Tw0))
                            qk[0] = self._leading_edge_q(
                                rho_inf=float(rho_inf),
                                v_inf=float(v_inf),
                                h0=float(h0),
                                h_w=float(h_w0),
                                alpha_rad=float(alpha_rad),
                                chi_rad=float(chi_rad),
                                rn_local_m=float(rn_local_m),
                            )
                        else:
                            qk[0] = windward_q_at_index(gas=self.gas, cache=cache, i=0, Tw_i=float(Tw0))
                    return np.asarray(qk, dtype=float)

                save_time = bool(cfg.get("save_time_history", False))
                if self.ny > 1 and save_time:
                    self._warn("transient_balance with ny>1: save_time_history forced to root strip only to avoid huge outputs.")

                if save_time and j == 0 and self.ny == 1:
                    Tw_time, q_time = march_explicit_balance(
                        Tw0=np.asarray(Tw0, dtype=float),
                        dt_s=float(dt),
                        n_steps=int(n_steps),
                        cap_J_per_m2K=float(cap),
                        emissivity=float(eps),
                        sigma_W_m2_K4=float(sigma),
                        Tw_min_K=float(Tw_min),
                        Tw_max_K=float(Tw_max),
                        eval_q_a=eval_q_a,
                    )
                    self.last_fields.update({"t_s": t_s, "Tw_w_time": Tw_time, "q_w_time": q_time})
                    Tw_w = Tw_time[-1, :].copy()
                    q_w = q_time[-1, :].copy()
                else:
                    Tw_w, q_w = march_explicit_balance_final(
                        Tw0=np.asarray(Tw0, dtype=float),
                        dt_s=float(dt),
                        n_steps=int(n_steps),
                        cap_J_per_m2K=float(cap),
                        emissivity=float(eps),
                        sigma_W_m2_K4=float(sigma),
                        Tw_min_K=float(Tw_min),
                        Tw_max_K=float(Tw_max),
                        eval_q_a=eval_q_a,
                    )

                # Leeward: keep baseline fixed-wall behavior for transient mode
                T_wall = float(self.case.wall_temperature_K if self.case.wall_temperature_K is not None else 300.0)
                q_l = self.calc_strip_heat_flux_fixed_wall(mach=mach, alpha_deg=alpha, chord_m=chord, side="leeward", T_wall_K=T_wall)
                Tw_l = np.full_like(q_l, T_wall, dtype=float)

            elif tw_type == "radiative_equilibrium":
                Tw_w, q_w = self.calc_strip_radiative_equilibrium(mach=mach, alpha_deg=alpha, chord_m=chord, side="windward")
                # couple leeward with windward enthalpy (baseline behavior)
                h_wwd = np.full((self.nx,), np.nan, dtype=float)
                for i in range(self.nx):
                    if np.isfinite(Tw_w[i]):
                        h_wwd[i] = float(self.gas.h_from_T(float(Tw_w[i])))
                p_inf, rho_inf, T_inf, v_inf = self._freestream(mach)
                h_inf = float(self.gas.h_from_T(T_inf))
                h0 = h_inf + 0.5 * (float(v_inf) ** 2)
                h_s = float(h0)
                ratio_T = normal_shock_temperature_ratio(gamma=float(self.case.gamma), mach=float(mach))
                T_ns = float(T_inf) * float(ratio_T)
                mu_ns = float(self.gas.mu(T_ns))
                Re_ns = leeward_re_ns(rho_inf=rho_inf, v_inf=v_inf, R_ref=self._leeward_R_ref(chord_m=chord), mu_ns=mu_ns)
                St_dist = leeward_stanton_distribution(Re_ns=float(Re_ns), h_wwd_dist=h_wwd, h_s=float(h_s))
                Tw_l, q_l = solve_leeward_radiative_equilibrium_coupled(
                    gas=self.gas,
                    rho_inf=rho_inf,
                    v_inf=v_inf,
                    St_dist=St_dist,
                    h_s=float(h_s),
                    emissivity=float(self.vehicle.emissivity),
                    sigma_W_m2_K4=float(self.case.sigma_W_m2_K4),
                )
            else:
                # fixed wall temperature model
                T_wall = float(self.case.wall_temperature_K if self.case.wall_temperature_K is not None else 300.0)
                q_w = self.calc_strip_heat_flux_fixed_wall(
                    mach=mach, alpha_deg=alpha, chord_m=chord, side="windward", T_wall_K=T_wall
                )
                q_l = self.calc_strip_heat_flux_fixed_wall(mach=mach, alpha_deg=alpha, chord_m=chord, side="leeward", T_wall_K=T_wall)
                Tw_w = np.full_like(q_w, T_wall, dtype=float)
                Tw_l = np.full_like(q_l, T_wall, dtype=float)

            q_w_list.append(q_w)
            q_l_list.append(q_l)
            Tw_w_list.append(Tw_w)
            Tw_l_list.append(Tw_l)

            # Diagnostics: transition weighting and Reynolds ratio along windward surface.
            slope_arr = self._windward_slope(alpha)
            p_inf, rho_inf, T_inf, _v_inf = self._freestream(mach)
            cache = build_windward_edge_cache(
                gas=self.gas,
                lf_cfg=self.lf_cfg,
                mach=float(mach),
                alpha_deg=float(alpha),
                sweep_le_deg=float(self.vehicle.sweep_le_deg),
                p_inf=float(p_inf),
                rho_inf=float(rho_inf),
                T_inf=float(T_inf),
                chord_m=float(chord),
                xc_grid=np.asarray(self.xc_grid, dtype=float),
                slope_arr=np.asarray(slope_arr, dtype=float),
                transition_x_over_c=self.case.transition_x_over_c,
            )
            re_edge = np.zeros((self.nx,), dtype=float)
            re_tri = np.zeros((self.nx,), dtype=float)
            w_tr = np.zeros((self.nx,), dtype=float)
            for i in range(self.nx):
                edge = cache.edges[i]
                re_edge[i] = float(edge.rho_e) * float(edge.v_e) * float(cache.x_phys[i]) / float(edge.mu_e)
                re_tri[i] = float(transition_reynolds(ma_e=float(edge.ma_e)))
                if i == 0:
                    w_tr[i] = 0.0
                    continue
                Tw_i = float(Tw_w[i])
                if not np.isfinite(Tw_i):
                    w_tr[i] = float("nan")
                    continue
                h_w = float(self.gas.h_from_T(Tw_i))
                _q_lam, _q_turb, re_x_star_lam, _re_x_star_turb = windward_ref_enthalpy_branches(
                    gas=self.gas,
                    edge=edge,
                    x=float(cache.x_phys[i]),
                    h_w=h_w,
                )
                w_tr[i] = float(
                    transition_weight(
                        enable=bool(self.lf_cfg.transition.enable),
                        re_measure=float(re_x_star_lam),
                        re_tri=float(re_tri[i]),
                        weighting=str(self.lf_cfg.transition.weighting),
                        width_decades=float(self.lf_cfg.transition.width_decades),
                        x_over_c=float(cache.x_over_c[i]),
                        transition_x_over_c=self.case.transition_x_over_c,
                    )
                )
            w_tr_list.append(w_tr)
            re_edge_list.append(re_edge)
            re_tri_list.append(re_tri)

        q_w_arr = np.array(q_w_list, dtype=float).reshape(-1)
        q_l_arr = np.array(q_l_list, dtype=float).reshape(-1)
        Tw_w_arr = np.array(Tw_w_list, dtype=float).reshape(-1)
        Tw_l_arr = np.array(Tw_l_list, dtype=float).reshape(-1)
        w_tr_arr = np.array(w_tr_list, dtype=float).reshape(-1)
        re_edge_arr = np.array(re_edge_list, dtype=float).reshape(-1)
        re_tri_arr = np.array(re_tri_list, dtype=float).reshape(-1)

        self.last_fields.update(
            {
                "q_w": q_w_arr,
                "q_l": q_l_arr,
                "Tw_w": Tw_w_arr,
                "Tw_l": Tw_l_arr,
                "w_tr": w_tr_arr,
                "re_edge": re_edge_arr,
                "re_tri": re_tri_arr,
            }
        )

        chunks = []
        for name in list(self.sampling.concat_order):
            if name not in self.last_fields:
                raise KeyError(f"Requested output field {name!r} not available. Available={list(self.last_fields.keys())}")
            chunks.append(np.asarray(self.last_fields[name], dtype=float).reshape(-1))
        return np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

