"""Faceted 3D branch solver (strong-3D minimal upgrade).

Design goals (per 新翼型.md):
- Do not contaminate the baseline 2D solver in `solver.py`
- Reuse the same physical "soup" (gas, atmosphere, heatflux, outputs)
- Replace ONLY the windward edge-chain angle definition (phi) to depend on (sx, sy)
- Apply a planform mask (triangle) for a lifting-body half planform
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .aero.edge_conditions import effective_alpha
from .aero.windward_cache_faceted3d import (
    build_windward_edge_cache_faceted3d,
    windward_q_at_index,
    windward_q_distribution_from_Tw,
)
from .aero.transition import transition_reynolds, transition_weight
from .config.lf_qw import LfQwConfig
from .gas.thermo import make_perfect_gas_thermo
from .gas.transport import mu_sutherland
from .geometry.faceted3d import (
    Faceted3DConfig,
    load_outline_csv,
    outline_strip_xle_chord_mask,
    triangle_strip_xle_chord_mask,
)
from .geometry.stl_surface import AsciiStlMesh, SurfaceSlopeSampler
from .heatflux.leading_edge import leading_edge_heat_flux_baseline
from .heatflux.leeward import (
    leeward_heat_flux_distribution,
    leeward_re_ns,
    leeward_stanton_distribution,
    normal_shock_temperature_ratio,
)
from .heatflux.windward import windward_ref_enthalpy_branches
from .sampling.grid import make_sampling_grids
from .specs.loader import load_yaml
from .specs.models import CaseSpec, SamplingSpec, VehicleSpec
from .thermal.leeward_equilibrium import solve_leeward_radiative_equilibrium_coupled
from .thermal.transient import march_explicit_balance, march_explicit_balance_final, require_transient_material
from .thermal.windward_equilibrium import solve_windward_radiative_equilibrium
from .types import GasModel
from .utils.warnings import WarningLog


class WingLowFidelitySolverFaceted3D:
    """Baseline-compatible solver façade, faceted 3D windward branch."""

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

        # Faceted3D config (required for this solver branch)
        vs = self.veh_spec_raw.get("vehicle_spec", {}) if isinstance(self.veh_spec_raw, dict) else {}
        f3 = vs.get("faceted3d", {}) if isinstance(vs, dict) else {}
        self.f3_cfg = Faceted3DConfig.from_faceted3d_spec(dict(f3))

        # Optional: load planform outline CSV (most robust geometry source).
        self.outline_x_m: np.ndarray | None = None
        self.outline_span_m: np.ndarray | None = None
        self.slope_sampler: SurfaceSlopeSampler | None = None
        self.planform_b_half_m: float = float(self.vehicle.b_half_m)
        if self.f3_cfg.outline_csv_path:
            p = (self.veh_path.parent / str(self.f3_cfg.outline_csv_path)).resolve()
            ox, oy = load_outline_csv(
                csv_path=p,
                x_col=str(self.f3_cfg.outline_x_col),
                span_col=str(self.f3_cfg.outline_span_col),
                span_sign=float(self.f3_cfg.outline_span_sign),
            )
            # Use non-negative span only for half-planform (we can mirror externally).
            span = np.asarray(oy, dtype=float)
            if np.nanmax(span) < 0:
                span = -span
            span = np.abs(span)
            self.outline_x_m = np.asarray(ox, dtype=float)
            self.outline_span_m = np.asarray(span, dtype=float)
            b_half_outline = float(np.nanmax(self.outline_span_m))
            if np.isfinite(b_half_outline) and b_half_outline > 0:
                self.planform_b_half_m = b_half_outline
                # Warn if mismatch is large (helps catch wrong units)
                b_spec = float(self.vehicle.b_half_m)
                if np.isfinite(b_spec) and b_spec > 0:
                    rel = abs(b_half_outline - b_spec) / b_spec
                    if rel > 0.05:
                        self._warn(
                            f"planform b_half mismatch: outline={b_half_outline:.6g} m vs spec={b_spec:.6g} m (rel={rel:.2%}). "
                            f"Using outline value for sampling."
                        )

        # Optional: load ASCII STL surface for local slope sampling.
        if self.f3_cfg.surface_stl_path:
            stl_path = (self.veh_path.parent / str(self.f3_cfg.surface_stl_path)).resolve()
            mesh = AsciiStlMesh.load(
                stl_path=stl_path,
                unit=str(self.f3_cfg.stl_unit),
                span_sign=float(self.f3_cfg.stl_span_sign),
                right_half_only=bool(self.f3_cfg.stl_right_half_only),
            )
            self.slope_sampler = SurfaceSlopeSampler(mesh=mesh)

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

        # Default facet slopes (constants); if STL is provided, these can be overridden per-point.
        self.sx_up, self.sy_up, self.sx_lo, self.sy_lo = self.f3_cfg.slopes()

        # Public result cache (baseline-style)
        self.last_fields: dict[str, Any] = {}

    def _warn(self, msg: str) -> None:
        self.warning_log.warn(msg)

    @staticmethod
    def _resolve_project_root() -> Path:
        here = Path(__file__).resolve()
        if len(here.parents) >= 3:
            return here.parents[2]
        return here.parent

    def _freestream(self, mach: float) -> tuple[float, float, float, float]:
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

    def _rn_local(self, *, chord_m: float) -> float:
        return float(self.vehicle.rn_m) * float(chord_m) / float(self.vehicle.c_root_m)

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
        # Mirror solver.py helper to keep baseline compatibility.
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

    def _leeward_R_ref(self, *, chord_m: float) -> float:
        rn = float(self.vehicle.rn_m)
        if rn > 0.0:
            return rn
        return max(float(chord_m), 1e-6)

    def _strip_xle_chord_mask(self, *, y_over_b: float) -> tuple[float, float, np.ndarray]:
        if self.outline_x_m is not None and self.outline_span_m is not None:
            return outline_strip_xle_chord_mask(
                x_over_c=np.asarray(self.xc_grid, dtype=float),
                y_over_b=float(y_over_b),
                b_half_m=float(self.planform_b_half_m),
                outline_x_m=np.asarray(self.outline_x_m, dtype=float),
                outline_span_m=np.asarray(self.outline_span_m, dtype=float),
                chord_min_m=float(self.f3_cfg.chord_min_m),
            )
        return triangle_strip_xle_chord_mask(
            x_over_c=np.asarray(self.xc_grid, dtype=float),
            y_over_b=float(y_over_b),
            c_root_m=float(self.vehicle.c_root_m),
            b_half_m=float(self.vehicle.b_half_m),
            half_angle_deg=float(self.f3_cfg.planform_half_angle_deg),
            chord_min_m=float(self.f3_cfg.chord_min_m),
        )

    def calc_strip_heat_flux_fixed_wall(
        self,
        *,
        mach: float,
        alpha_deg: float,
        chord_m: float,
        side: str,
        T_wall_K: float,
        sx_arr: np.ndarray,
        sy_arr: np.ndarray,
    ) -> np.ndarray:
        """Compute q(x) for one strip with fixed wall temperature."""

        if side not in {"windward", "leeward"}:
            raise ValueError(f"Invalid side={side!r}")

        p_inf, rho_inf, T_inf, v_inf = self._freestream(mach)
        h_inf = float(self.gas.h_from_T(T_inf))
        h0 = h_inf + 0.5 * (v_inf**2)
        h_w = float(self.gas.h_from_T(float(T_wall_K)))

        # Leeward is a mean correlation and does not depend on (sx, sy) here.
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

        # Windward (uses 3D edge cache)
        # Precompute phi clamp warning preview to match baseline behavior.
        phi_clamp_enable = bool(self.lf_cfg.phi_clamp.enable)
        phi_warn = bool(self.lf_cfg.phi_clamp.warn)
        phi_min = float(self.lf_cfg.phi_clamp.phi_min_rad)
        chi_rad = float(np.deg2rad(self.vehicle.sweep_le_deg))
        alpha_rad = float(np.deg2rad(alpha_deg))
        alpha_e = float(effective_alpha(alpha_rad, chi_rad))
        # phi = asin( -u·n_hat ) where n_hat uses (sx, sy)
        denom = np.sqrt(1.0 + np.asarray(sx_arr, dtype=float) ** 2 + np.asarray(sy_arr, dtype=float) ** 2)
        denom = np.where(denom <= 0.0, 1.0, denom)
        s = (np.sin(alpha_e) - np.asarray(sx_arr, dtype=float) * np.cos(alpha_e)) / denom
        s = np.clip(s, -1.0, 1.0)
        phi_arr = np.arcsin(s)
        clamped_idx = np.where(phi_clamp_enable & (phi_arr <= phi_min))[0]
        if phi_warn and clamped_idx.size > 0:
            preview = ", ".join([f"{float(self.xc_grid[i]):.4f}" for i in clamped_idx[:6].tolist()])
            self._warn(
                f"phi clamped at {int(clamped_idx.size)} points | M={mach:.2f}, alpha={alpha_deg:.2f}, side=windward | first x/c: [{preview}]"
            )

        cache = build_windward_edge_cache_faceted3d(
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
            sx_arr=np.asarray(sx_arr, dtype=float),
            sy_arr=np.asarray(sy_arr, dtype=float),
            transition_x_over_c=self.case.transition_x_over_c,
        )

        Tw = np.full((self.nx,), float(T_wall_K), dtype=float)
        q_dist = windward_q_distribution_from_Tw(gas=self.gas, cache=cache, Tw=Tw)

        # Leading edge (index 0)
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
        return q_dist

    def calc_strip_radiative_equilibrium(
        self,
        *,
        mach: float,
        alpha_deg: float,
        chord_m: float,
        sx_arr: np.ndarray,
        sy_arr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute (Tw(x), q(x)) using steady radiative equilibrium (windward only)."""

        p_inf, rho_inf, T_inf, v_inf = self._freestream(mach)
        h_inf = float(self.gas.h_from_T(T_inf))
        h0 = h_inf + 0.5 * (float(v_inf) ** 2)
        chi_rad = float(np.deg2rad(self.vehicle.sweep_le_deg))
        alpha_rad = float(np.deg2rad(alpha_deg))

        cache = build_windward_edge_cache_faceted3d(
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
            sx_arr=np.asarray(sx_arr, dtype=float),
            sy_arr=np.asarray(sy_arr, dtype=float),
            transition_x_over_c=self.case.transition_x_over_c,
        )
        rn_local_m = self._rn_local(chord_m=chord_m)

        def q_leading_edge_of_Tw(Tw0: float) -> float:
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
            cache=cache,  # type: ignore[arg-type]
            emissivity=float(self.vehicle.emissivity),
            sigma_W_m2_K4=float(self.case.sigma_W_m2_K4),
            q_leading_edge_of_Tw=q_leading_edge_of_Tw,
        )

    def compute_snapshot(self, *, mach: float, alpha: float) -> np.ndarray:
        self.last_fields = {}
        tw_type = str(self.case.tw_model_type or "").strip().lower()

        q_w_list = []
        q_l_list = []
        Tw_w_list = []
        Tw_l_list = []
        w_tr_list = []
        re_edge_list = []
        re_tri_list = []

        # Choose windward/leeward facet slopes by alpha sign (mirror baseline convention)
        use_stl = self.slope_sampler is not None

        for j in range(self.ny):
            yb = float(self.yb_grid[j])
            x_le, chord, mask_x = self._strip_xle_chord_mask(y_over_b=yb)
            mask_x = np.asarray(mask_x, dtype=bool).reshape(-1)
            if mask_x.size != self.nx:
                raise ValueError("triangle mask must have length nx")

            # Default invalid strip: fill NaNs and continue
            if not np.any(mask_x) or not (float(chord) > 0.0) or not np.isfinite(float(x_le)):
                q_w = np.full((self.nx,), float("nan"), dtype=float)
                q_l = np.full((self.nx,), float("nan"), dtype=float)
                Tw_w = np.full((self.nx,), float("nan"), dtype=float)
                Tw_l = np.full((self.nx,), float("nan"), dtype=float)
                w_tr = np.full((self.nx,), float("nan"), dtype=float)
                re_edge = np.full((self.nx,), float("nan"), dtype=float)
                re_tri = np.full((self.nx,), float("nan"), dtype=float)
                q_w_list.append(q_w)
                q_l_list.append(q_l)
                Tw_w_list.append(Tw_w)
                Tw_l_list.append(Tw_l)
                w_tr_list.append(w_tr)
                re_edge_list.append(re_edge)
                re_tri_list.append(re_tri)
                continue

            # Build (sx, sy) arrays for this strip.
            if use_stl:
                assert self.slope_sampler is not None
                span_m = float(yb) * float(self.planform_b_half_m)
                x_pts = float(x_le) + np.asarray(self.xc_grid, dtype=float) * float(chord)
                sx_up_arr = np.full((self.nx,), np.nan, dtype=float)
                sy_up_arr = np.full((self.nx,), np.nan, dtype=float)
                sx_lo_arr = np.full((self.nx,), np.nan, dtype=float)
                sy_lo_arr = np.full((self.nx,), np.nan, dtype=float)
                for i in range(self.nx):
                    if not bool(mask_x[i]):
                        continue
                    up_s, lo_s = self.slope_sampler.sample_upper_lower(x=float(x_pts[i]), span=float(span_m))
                    if up_s is not None:
                        sx_up_arr[i], sy_up_arr[i] = float(up_s[0]), float(up_s[1])
                    if lo_s is not None:
                        sx_lo_arr[i], sy_lo_arr[i] = float(lo_s[0]), float(lo_s[1])
                sx_up_arr = np.where(np.isfinite(sx_up_arr), sx_up_arr, float(self.sx_up))
                sy_up_arr = np.where(np.isfinite(sy_up_arr), sy_up_arr, float(self.sy_up))
                sx_lo_arr = np.where(np.isfinite(sx_lo_arr), sx_lo_arr, float(self.sx_lo))
                sy_lo_arr = np.where(np.isfinite(sy_lo_arr), sy_lo_arr, float(self.sy_lo))
            else:
                sx_up_arr = np.full((self.nx,), float(self.sx_up), dtype=float)
                sy_up_arr = np.full((self.nx,), float(self.sy_up), dtype=float)
                sx_lo_arr = np.full((self.nx,), float(self.sx_lo), dtype=float)
                sy_lo_arr = np.full((self.nx,), float(self.sy_lo), dtype=float)

            # Choose windward/leeward by alpha sign (baseline convention).
            if float(alpha) >= 0.0:
                sx_w_arr, sy_w_arr = sx_lo_arr, sy_lo_arr
                sx_l_arr, sy_l_arr = sx_up_arr, sy_up_arr
            else:
                sx_w_arr, sy_w_arr = sx_up_arr, sy_up_arr
                sx_l_arr, sy_l_arr = sx_lo_arr, sy_lo_arr

            if tw_type == "transient_balance":
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

                # Precompute cache for eval_q_a
                p_inf, rho_inf, T_inf, v_inf = self._freestream(mach)
                cache = build_windward_edge_cache_faceted3d(
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
                    sx_arr=sx_w_arr,
                    sy_arr=sy_w_arr,
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
                    Tw0i = float(Tw_k[0]) if Tw_k.size > 0 else float("nan")
                    if np.isfinite(Tw0i):
                        if float(self.vehicle.rn_m) > 0.0:
                            h_w0 = float(self.gas.h_from_T(Tw0i))
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
                            qk[0] = windward_q_at_index(gas=self.gas, cache=cache, i=0, Tw_i=float(Tw0i))
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

                # Leeward: baseline fixed-wall behavior for transient mode
                T_wall = float(self.case.wall_temperature_K if self.case.wall_temperature_K is not None else 300.0)
                q_l = self.calc_strip_heat_flux_fixed_wall(
                    mach=mach,
                    alpha_deg=alpha,
                    chord_m=chord,
                    side="leeward",
                    T_wall_K=T_wall,
                    sx_arr=sx_l_arr,
                    sy_arr=sy_l_arr,
                )
                Tw_l = np.full_like(q_l, T_wall, dtype=float)

            elif tw_type == "radiative_equilibrium":
                Tw_w, q_w = self.calc_strip_radiative_equilibrium(
                    mach=mach, alpha_deg=alpha, chord_m=chord, sx_arr=sx_w_arr, sy_arr=sy_w_arr
                )
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
                Re_ns = leeward_re_ns(
                    rho_inf=rho_inf, v_inf=v_inf, R_ref=self._leeward_R_ref(chord_m=chord), mu_ns=mu_ns
                )
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
                    mach=mach,
                    alpha_deg=alpha,
                    chord_m=chord,
                    side="windward",
                    T_wall_K=T_wall,
                    sx_arr=sx_w_arr,
                    sy_arr=sy_w_arr,
                )
                q_l = self.calc_strip_heat_flux_fixed_wall(
                    mach=mach,
                    alpha_deg=alpha,
                    chord_m=chord,
                    side="leeward",
                    T_wall_K=T_wall,
                    sx_arr=sx_l_arr,
                    sy_arr=sy_l_arr,
                )
                Tw_w = np.full_like(q_w, T_wall, dtype=float)
                Tw_l = np.full_like(q_l, T_wall, dtype=float)

            # Apply planform mask
            q_w = np.asarray(q_w, dtype=float)
            q_l = np.asarray(q_l, dtype=float)
            Tw_w = np.asarray(Tw_w, dtype=float)
            Tw_l = np.asarray(Tw_l, dtype=float)
            q_w = np.where(mask_x, q_w, np.nan)
            q_l = np.where(mask_x, q_l, np.nan)
            Tw_w = np.where(mask_x, Tw_w, np.nan)
            Tw_l = np.where(mask_x, Tw_l, np.nan)

            q_w_list.append(q_w)
            q_l_list.append(q_l)
            Tw_w_list.append(Tw_w)
            Tw_l_list.append(Tw_l)

            # Diagnostics: transition weighting and Reynolds ratio along windward surface.
            p_inf, rho_inf, T_inf, _v_inf = self._freestream(mach)
            cache = build_windward_edge_cache_faceted3d(
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
                sx_arr=sx_w_arr,
                sy_arr=sy_w_arr,
                transition_x_over_c=self.case.transition_x_over_c,
            )
            re_edge = np.full((self.nx,), np.nan, dtype=float)
            re_tri = np.full((self.nx,), np.nan, dtype=float)
            w_tr = np.full((self.nx,), np.nan, dtype=float)
            for i in range(self.nx):
                if not bool(mask_x[i]):
                    continue
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

