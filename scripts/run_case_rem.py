#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run one case using the rewritten ref_enthalpy_method solver.

This script mirrors the baseline workflow:
- load vehicle/case/sampling specs
- run one (mach, alpha)
- write runs/<run_dir>/summary.json and fields.npz
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _ensure_import_path() -> Path:
    """Ensure `import ref_enthalpy_method` works for src-layout.

    Supports running from:
    - repo root:   python scripts/run_case_rem.py
    - scripts dir: python run_case_rem.py
    """

    here = Path(__file__).resolve()
    repo_root = here.parents[1]  # .../<repo>/scripts/run_case_rem.py
    src_root = repo_root / "src"
    # Add both repo_root and src_root for flexibility
    for p in (str(repo_root), str(src_root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return repo_root


def _np_to_builtin(x: Any) -> Any:
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    return x


def _summarize_array(name: str, arr: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(arr, dtype=float).reshape(-1)
    finite = np.isfinite(arr)
    out: dict[str, Any] = {
        "name": name,
        "shape": list(arr.shape),
        "finite_count": int(np.sum(finite)),
        "nan_count": int(np.sum(np.isnan(arr))),
        "inf_count": int(np.sum(~finite) - np.sum(np.isnan(arr))),
    }
    if np.any(finite):
        a = arr[finite]
        out.update({"min": float(np.min(a)), "max": float(np.max(a)), "mean": float(np.mean(a))})
    return out


def _profile_samples(xc: np.ndarray, arr: np.ndarray, n_points: int = 6) -> list[dict[str, Any]]:
    xc = np.asarray(xc, dtype=float).reshape(-1)
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if xc.size != arr.size or xc.size == 0:
        return []
    idx = np.unique(np.round(np.linspace(0, xc.size - 1, max(int(n_points), 2))).astype(int))
    out: list[dict[str, Any]] = []
    for i in idx.tolist():
        out.append({"i": int(i), "x_over_c": float(xc[i]), "value": _np_to_builtin(arr[i])})
    return out


def _monotonic_decrease_report(
    *,
    x_over_c: np.ndarray,
    y: np.ndarray,
    tol: float = 1.0,
    max_violations: int = 8,
) -> dict[str, Any]:
    """Check if y(x) is (approximately) monotonically decreasing.

    Returns a compact report of any increases (dy > tol).
    """

    x = np.asarray(x_over_c, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size or x.size < 2:
        return {"ok": False, "reason": "shape_mismatch_or_too_short"}

    dy = np.diff(y)
    inc_idx = np.where(dy > float(tol))[0]  # dy at i corresponds to segment i->i+1
    out: dict[str, Any] = {
        "ok": bool(inc_idx.size == 0),
        "tol": float(tol),
        "n": int(x.size),
        "n_increase_segments": int(inc_idx.size),
    }
    if inc_idx.size > 0:
        worst_i = int(inc_idx[np.argmax(dy[inc_idx])])
        out["worst_increase"] = {
            "i0": worst_i,
            "x0": float(x[worst_i]),
            "x1": float(x[worst_i + 1]),
            "y0": float(y[worst_i]),
            "y1": float(y[worst_i + 1]),
            "dy": float(dy[worst_i]),
        }
        segs: list[dict[str, Any]] = []
        for i in inc_idx[: max(int(max_violations), 1)].tolist():
            segs.append(
                {
                    "i0": int(i),
                    "x0": float(x[i]),
                    "x1": float(x[i + 1]),
                    "y0": float(y[i]),
                    "y1": float(y[i + 1]),
                    "dy": float(dy[i]),
                }
            )
        out["increase_segments_preview"] = segs
    return out


def _triangulate_structured(ny: int, nx: int) -> np.ndarray:
    """Triangulate a (ny,nx) structured grid for tricontourf."""
    triangles = np.empty((2 * (ny - 1) * (nx - 1), 3), dtype=np.int32)
    k = 0
    for j in range(ny - 1):
        row = j * nx
        row2 = (j + 1) * nx
        for i in range(nx - 1):
            p00 = row + i
            p01 = row + i + 1
            p10 = row2 + i
            p11 = row2 + i + 1
            triangles[k] = (p00, p01, p11)
            triangles[k + 1] = (p00, p11, p10)
            k += 2
    return triangles


def _try_save_plots(
    *,
    solver,
    out_dir: Path,
    fields: dict[str, np.ndarray],
    summary: dict[str, Any],
    plot_x_over_c_min: float = 0.01,
) -> list[str]:
    """Best-effort plotting. Returns created file paths (strings)."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
    except Exception:
        # Keep runs working even if matplotlib isn't installed.
        print("note: matplotlib not available; skipping plot generation.")
        return []

    created: list[str] = []

    # Determine sampling mode from resolved sampling spec (more reliable than guessing).
    mode = str(getattr(solver.sampling, "mode", "")).strip()
    x_over_c_min = max(float(plot_x_over_c_min), 0.0)

    def _levels_from_minmax(vmin: float, vmax: float, n: int = 40) -> np.ndarray:
        """Robust contour levels even for constant fields."""
        if not (np.isfinite(vmin) and np.isfinite(vmax)):
            raise ValueError("non-finite min/max")
        if vmax <= vmin:
            eps = max(1e-6, 1e-6 * max(abs(vmin), abs(vmax), 1.0))
            vmin = vmin - eps
            vmax = vmax + eps
        return np.linspace(vmin, vmax, int(n))

    # 1D root chord line plots
    if mode == "root_windward_chord_line" and int(getattr(solver, "ny", 0)) == 1:
        xc = np.asarray(getattr(solver, "xc_grid"), dtype=float).reshape(-1)
        c_root = float(getattr(solver.vehicle, "c_root_m"))
        x_m = xc * c_root
        mask = xc >= x_over_c_min

        for side, key, ylabel, prefix in (
            ("windward", "Tw_w", "T / K", "Tw_root_chord"),
            ("leeward", "Tw_l", "T / K", "Tw_root_chord"),
            ("windward", "q_w", "q / W/m^2", "q_root_chord"),
            ("leeward", "q_l", "q / W/m^2", "q_root_chord"),
        ):
            if key not in fields:
                continue
            Tw = np.asarray(fields[key], dtype=float).reshape(-1)
            if Tw.size != x_m.size:
                continue
            fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=170)
            ax.plot(x_m[mask], Tw[mask], linewidth=1.6)
            ax.set_xlabel("x / m")
            ax.set_ylabel(str(ylabel))
            meta_mach = summary.get("inputs", {}).get("mach", None)
            meta_alpha = summary.get("inputs", {}).get("alpha_deg", None)
            h_m = summary.get("freestream", {}).get("h_m", summary.get("inputs", {}).get("h_m_override", None))
            h_km = (None if h_m is None else float(h_m) / 1000.0)
            h_text = ("h=?" if h_km is None else f"h={h_km:.2f} km")
            ax.set_title(f"{key} root chord ({side}) | {h_text}, M={meta_mach}, alpha={meta_alpha} deg")
            ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
            fig.tight_layout()
            p = out_dir / f"{prefix}_{side}.png"
            fig.savefig(p)
            plt.close(fig)
            created.append(str(p))

    # 2D half-wing surface temperature plots
    if mode == "full_wing_surface_grid" and int(getattr(solver, "ny", 0)) > 1:
        nx = int(getattr(solver, "nx"))
        ny = int(getattr(solver, "ny"))
        xc = np.asarray(getattr(solver, "xc_grid"), dtype=float).reshape(-1)
        yb = np.asarray(getattr(solver, "yb_grid"), dtype=float).reshape(-1)
        if xc.size == nx and yb.size == ny:
            # For visual comparison, optionally drop a small leading segment (x/c < x_over_c_min).
            col_mask = xc >= x_over_c_min
            if not np.any(col_mask):
                col_mask = np.ones_like(xc, dtype=bool)
            xc_plot = xc[col_mask]
            nx_plot = int(xc_plot.size)

            # Plot geometry:
            # - Default: trapezoid planform from VehicleSpec (baseline 2D solver behavior)
            # - Faceted3D: use solver's own strip geometry (outline/STL driven) if available
            is_faceted3d = bool(hasattr(solver, "_strip_xle_chord_mask"))
            b_half = float(getattr(getattr(solver, "vehicle", object()), "b_half_m", 0.0))
            if is_faceted3d and hasattr(solver, "planform_b_half_m"):
                try:
                    b_half = float(getattr(solver, "planform_b_half_m"))
                except Exception:
                    pass

            c_root = float(getattr(solver.vehicle, "c_root_m"))
            c_tip = float(getattr(solver.vehicle, "c_tip_m"))
            sweep_deg = float(getattr(solver.vehicle, "sweep_le_deg"))
            chi = float(np.deg2rad(sweep_deg))

            X = np.zeros((ny, nx_plot), dtype=float)
            Y = np.zeros((ny, nx_plot), dtype=float)
            for j in range(ny):
                y = float(yb[j]) * float(b_half)
                Y[j, :] = y
                if is_faceted3d:
                    try:
                        x_le, chord, _mask = solver._strip_xle_chord_mask(y_over_b=float(yb[j]))  # noqa: SLF001
                        x_le = float(x_le)
                        chord = float(chord)
                        if not (np.isfinite(x_le) and np.isfinite(chord) and chord > 0.0):
                            # keep coordinates finite; z-values will be masked out anyway
                            x_le, chord = 0.0, float(c_root)
                        X[j, :] = x_le + xc_plot * chord
                        continue
                    except Exception:
                        # fall back to baseline geometry
                        pass
                chord = float(c_root + (c_tip - c_root) * float(yb[j]))
                x_le = float(y) * float(np.tan(chi))
                X[j, :] = x_le + xc_plot * chord

            triangles = _triangulate_structured(ny, nx_plot)
            tri_x = X.reshape(-1)
            tri_y = Y.reshape(-1)

            for side, key, cbar_label, prefix in (
                ("windward", "Tw_w", "T / K", "Tw_surface"),
                ("leeward", "Tw_l", "T / K", "Tw_surface"),
                ("windward", "q_w", "q / W/m^2", "q_surface"),
                ("leeward", "q_l", "q / W/m^2", "q_surface"),
            ):
                if key not in fields:
                    continue
                z = np.asarray(fields[key], dtype=float).reshape(-1)
                if z.size != nx * ny:
                    continue
                z2 = z.reshape(ny, nx)[:, col_mask].reshape(-1)
                finite = np.isfinite(z2)
                if not np.any(finite):
                    continue

                # Matplotlib's tricontourf does not allow non-finite z at vertices
                # within the triangulation. Since we purposely set planform-outside
                # points to NaN, mask any triangle that touches a non-finite vertex.
                tri_mask = np.any(~finite[triangles], axis=1)
                tri = mtri.Triangulation(tri_x, tri_y, triangles, mask=tri_mask)

                vmin = float(np.nanmin(z2))
                vmax = float(np.nanmax(z2))
                if not (np.isfinite(vmin) and np.isfinite(vmax)):
                    continue

                fig, ax = plt.subplots(figsize=(7.8, 4.6), dpi=170)
                levels = _levels_from_minmax(vmin=vmin, vmax=vmax, n=40)
                im = ax.tricontourf(tri, z2, levels=levels, cmap="turbo")
                cbar = fig.colorbar(im, ax=ax, pad=0.02)
                cbar.set_label(str(cbar_label))

                # planform outline
                if is_faceted3d and hasattr(solver, "outline_x_m") and hasattr(solver, "outline_span_m"):
                    try:
                        ox = np.asarray(getattr(solver, "outline_x_m"), dtype=float).reshape(-1)
                        oy = np.asarray(getattr(solver, "outline_span_m"), dtype=float).reshape(-1)
                        ok = np.isfinite(ox) & np.isfinite(oy)
                        ox = ox[ok]
                        oy = oy[ok]
                        if ox.size >= 3:
                            # ensure closed polyline for display
                            if not (abs(float(ox[0]) - float(ox[-1])) < 1e-9 and abs(float(oy[0]) - float(oy[-1])) < 1e-9):
                                ox = np.concatenate([ox, ox[:1]])
                                oy = np.concatenate([oy, oy[:1]])
                            ax.plot(ox, oy, color="k", linewidth=1.0, alpha=0.85)
                    except Exception:
                        pass
                else:
                    y0 = 0.0
                    y1 = float(b_half)
                    x_le_root = 0.0
                    x_le_tip = y1 * float(np.tan(chi))
                    x_te_root = x_le_root + float(c_root)
                    x_te_tip = x_le_tip + float(c_tip)
                    ax.plot(
                        [x_le_root, x_le_tip, x_te_tip, x_te_root, x_le_root],
                        [y0, y1, y1, y0, y0],
                        color="k",
                        linewidth=1.0,
                        alpha=0.85,
                    )

                meta_mach = summary.get("inputs", {}).get("mach", None)
                meta_alpha = summary.get("inputs", {}).get("alpha_deg", None)
                h_m = summary.get("freestream", {}).get("h_m", summary.get("inputs", {}).get("h_m_override", None))
                h_km = (None if h_m is None else float(h_m) / 1000.0)
                h_text = ("h=?" if h_km is None else f"h={h_km:.2f} km")
                ax.set_title(f"{key} surface ({side}) | {h_text}, M={meta_mach}, alpha={meta_alpha} deg")
                ax.set_xlabel("x / m")
                ax.set_ylabel("y / m (half-span)")
                ax.set_aspect("equal", adjustable="box")
                ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
                fig.tight_layout()

                p = out_dir / f"{prefix}_{side}.png"
                fig.savefig(p)
                plt.close(fig)
                created.append(str(p))

    return created


def main() -> int:
    _ensure_import_path()

    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicle", default="specs/vehicles/trapezoid_doublewedge_t0p034_sweep34.yaml")
    ap.add_argument("--case", default="specs/cases/doc_ma5_alpha0_15_h30km.yaml")
    ap.add_argument("--sampling", default="specs/sampling/engineering_full_wing_surface_grid_81x41.yaml")
    ap.add_argument("--run_dir", default="runs/trap_dw_t0034_ma5_h30km_a0_2d")
    ap.add_argument("--mach", type=float, default=5.0)
    ap.add_argument("--alpha", type=float, default=0.0, help="deg")
    ap.add_argument("--h_m", type=float, default=None, help="override flight altitude (meters)")
    ap.add_argument("--h_km", type=float, default=None, help="override flight altitude (kilometers)")
    ap.add_argument("--save_npz", action="store_true", default=True)
    ap.add_argument("--no_plots", action="store_true", help="do not generate plot images")
    ap.add_argument("--plot_x_over_c_min", type=float, default=0.002, help="plot only x/c >= this (default: 0.01)")
    args = ap.parse_args()

    # Select solver branch by vehicle spec content:
    # - default: 2D/strip-theory solver (`ref_enthalpy_method.solver.WingLowFidelitySolver`)
    # - faceted3d: enhanced solver with 3D facet normals (`ref_enthalpy_method.solver_faceted3d.WingLowFidelitySolverFaceted3D`)
    from ref_enthalpy_method.specs.loader import load_yaml  # runtime import

    veh_root = load_yaml(str(args.vehicle))
    veh_spec = veh_root.get("vehicle_spec", {}) if isinstance(veh_root, dict) else {}
    use_faceted3d = bool(isinstance(veh_spec, dict) and ("faceted3d" in veh_spec))

    if use_faceted3d:
        from ref_enthalpy_method.solver_faceted3d import WingLowFidelitySolverFaceted3D  # runtime import

        solver = WingLowFidelitySolverFaceted3D(
            vehicle_config=str(args.vehicle),
            case_config=str(args.case),
            sampling_config=str(args.sampling),
            run_dir=str(args.run_dir),
        )
    else:
        from ref_enthalpy_method.solver import WingLowFidelitySolver  # runtime import

        solver = WingLowFidelitySolver(
            vehicle_config=str(args.vehicle),
            case_config=str(args.case),
            sampling_config=str(args.sampling),
            run_dir=str(args.run_dir),
        )

    # Optional override: flight altitude.
    h_override = None
    if args.h_m is not None and args.h_km is not None:
        raise ValueError("Use only one of --h_m or --h_km")
    if args.h_m is not None:
        h_override = float(args.h_m)
    if args.h_km is not None:
        h_override = float(args.h_km) * 1000.0
    if h_override is not None:
        solver.case = replace(solver.case, fixed_h_m=float(h_override))

    _u = solver.compute_snapshot(mach=float(args.mach), alpha=float(args.alpha))
    fields = dict(solver.last_fields or {})

    out_dir = Path(solver.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "inputs": {
            "vehicle_config": str(args.vehicle),
            "case_config": str(args.case),
            "sampling_config": str(args.sampling),
            "run_dir": str(args.run_dir),
            "mach": float(args.mach),
            "alpha_deg": float(args.alpha),
            "h_m_override": (None if h_override is None else float(h_override)),
        },
        "resolved_paths": {
            "project_root": str(solver.project_root),
            "vehicle_path": str(solver.veh_path),
            "case_path": str(solver.case_path),
            "sampling_path": str(solver.samp_path),
        },
        "vehicle": _np_to_builtin(solver.vehicle.__dict__),
        "case": _np_to_builtin(solver.case.__dict__),
        "sampling": _np_to_builtin(solver.sampling.__dict__),
        "outputs_available": list(fields.keys()),
    }
    # Warning summary
    try:
        warnings = list(getattr(solver, "warning_log").warnings)  # type: ignore[attr-defined]
        summary["warnings_count"] = int(len(warnings))
        summary["warnings_preview"] = warnings[:10]
        summary["warnings_log_path"] = str(Path(solver.run_dir) / "lf_warnings.log")
    except Exception:
        pass

    # Freestream summary (baseline-style)
    try:
        p_inf, rho_inf, T_inf, v_inf = solver._freestream(float(args.mach))  # noqa: SLF001
        summary["freestream"] = {
            "h_m": float(solver.case.fixed_h_m),
            "p_inf_Pa": float(p_inf),
            "rho_inf_kg_m3": float(rho_inf),
            "T_inf_K": float(T_inf),
            "gamma": float(solver.case.gamma),
            "R_J_per_kgK": float(solver.case.R_J_per_kgK),
            "V_inf_m_s": float(v_inf),
            "atmosphere_model": str(solver.case.atmosphere_model),
        }
    except Exception as e:
        summary["freestream_error"] = str(e)

    arrays_to_save: dict[str, np.ndarray] = {}
    for key in ("q_w", "q_l", "Tw_w", "Tw_l", "w_tr", "re_edge", "re_tri", "t_s", "Tw_w_time", "q_w_time"):
        if key in fields:
            arrays_to_save[key] = np.asarray(fields[key], dtype=float)
            summary[f"summary_{key}"] = _summarize_array(key, arrays_to_save[key])

    # Transition diagnostics
    if "w_tr" in arrays_to_save:
        w_tr = np.asarray(arrays_to_save["w_tr"], dtype=float).reshape(-1)
        finite = np.isfinite(w_tr)
        if np.any(finite):
            wf = w_tr[finite]
            summary["transition_stats"] = {
                "w_tr_max": float(np.max(wf)),
                "w_tr_mean": float(np.mean(wf)),
                "w_tr_frac_gt_0p5": float(np.mean(wf > 0.5)),
                "w_tr_frac_gt_0p1": float(np.mean(wf > 0.1)),
            }

    # Quick distribution samples for 1D root line runs
    if hasattr(solver, "xc_grid"):
        xc = np.asarray(solver.xc_grid, dtype=float).reshape(-1)
        if "q_w" in arrays_to_save and arrays_to_save["q_w"].reshape(-1).size == xc.size:
            summary["samples_q_w"] = _profile_samples(xc, arrays_to_save["q_w"])
        if "Tw_w" in arrays_to_save and arrays_to_save["Tw_w"].reshape(-1).size == xc.size:
            summary["samples_Tw_w"] = _profile_samples(xc, arrays_to_save["Tw_w"])
        if "q_l" in arrays_to_save and arrays_to_save["q_l"].reshape(-1).size == xc.size:
            summary["samples_q_l"] = _profile_samples(xc, arrays_to_save["q_l"])
        if "Tw_l" in arrays_to_save and arrays_to_save["Tw_l"].reshape(-1).size == xc.size:
            summary["samples_Tw_l"] = _profile_samples(xc, arrays_to_save["Tw_l"])

        # 2D mode diagnostics: chordwise monotonicity (root strip + spanwise mean).
        mode = str(getattr(solver.sampling, "mode", "")).strip()
        if mode == "full_wing_surface_grid":
            nx = int(getattr(solver, "nx", xc.size))
            ny = int(getattr(solver, "ny", 0))
            if nx > 1 and ny > 1:
                def _reshape2(a: np.ndarray) -> np.ndarray:
                    return np.asarray(a, dtype=float).reshape(ny, nx)

                if "Tw_w" in arrays_to_save and arrays_to_save["Tw_w"].size == nx * ny:
                    Tw2 = _reshape2(arrays_to_save["Tw_w"])
                    Tw_root = Tw2[0, :]
                    Tw_mean_y = np.nanmean(Tw2, axis=0)
                    summary["diagnostics_Tw_w_chordwise"] = {
                        "root_strip": {
                            "monotone_decrease_tol_1K": _monotonic_decrease_report(x_over_c=xc, y=Tw_root, tol=1.0),
                            "samples": _profile_samples(xc, Tw_root, n_points=10),
                        },
                        "spanwise_mean": {
                            "monotone_decrease_tol_1K": _monotonic_decrease_report(x_over_c=xc, y=Tw_mean_y, tol=1.0),
                            "samples": _profile_samples(xc, Tw_mean_y, n_points=10),
                        },
                    }

                # Also report w_tr chordwise trend so we can correlate with Tw rise.
                if "w_tr" in arrays_to_save and arrays_to_save["w_tr"].size == nx * ny:
                    w2 = _reshape2(arrays_to_save["w_tr"])
                    w_root = w2[0, :]
                    w_mean_y = np.nanmean(w2, axis=0)
                    summary["diagnostics_w_tr_chordwise"] = {
                        "root_strip": {"samples": _profile_samples(xc, w_root, n_points=10)},
                        "spanwise_mean": {"samples": _profile_samples(xc, w_mean_y, n_points=10)},
                    }

    # Radiative equilibrium residual check: q_a - eps*sigma*Tw^4 (should be ~0)
    eps = float(getattr(solver.vehicle, "emissivity", float("nan")))
    sigma = float(getattr(solver.case, "sigma_W_m2_K4", float("nan")))
    if np.isfinite(eps) and np.isfinite(sigma):
        if ("q_w" in arrays_to_save) and ("Tw_w" in arrays_to_save):
            q = np.asarray(arrays_to_save["q_w"], dtype=float).reshape(-1)
            Tw = np.asarray(arrays_to_save["Tw_w"], dtype=float).reshape(-1)
            if q.size == Tw.size:
                res = q - eps * sigma * (Tw**4)
                summary["radiative_balance_windward"] = {
                    "eps": eps,
                    "sigma": sigma,
                    "residual_abs_max": float(np.nanmax(np.abs(res))),
                    "residual_abs_mean": float(np.nanmean(np.abs(res))),
                    "residual_rel_max": float(np.nanmax(np.abs(res) / np.maximum(1.0, np.abs(q)))),
                }
        if ("q_l" in arrays_to_save) and ("Tw_l" in arrays_to_save):
            q = np.asarray(arrays_to_save["q_l"], dtype=float).reshape(-1)
            Tw = np.asarray(arrays_to_save["Tw_l"], dtype=float).reshape(-1)
            if q.size == Tw.size:
                res = q - eps * sigma * (Tw**4)
                summary["radiative_balance_leeward"] = {
                    "eps": eps,
                    "sigma": sigma,
                    "residual_abs_max": float(np.nanmax(np.abs(res))),
                    "residual_abs_mean": float(np.nanmean(np.abs(res))),
                    "residual_rel_max": float(np.nanmax(np.abs(res) / np.maximum(1.0, np.abs(q)))),
                }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if bool(args.save_npz) and len(arrays_to_save) > 0:
        np.savez_compressed(out_dir / "fields.npz", **arrays_to_save)

    created_plots: list[str] = []
    if not bool(args.no_plots):
        created_plots = _try_save_plots(
            solver=solver,
            out_dir=out_dir,
            fields=arrays_to_save,
            summary=summary,
            plot_x_over_c_min=float(args.plot_x_over_c_min),
        )

    print("=== ref_enthalpy_method run_case ===")
    print(f"written: {out_dir / 'summary.json'}")
    if bool(args.save_npz):
        print(f"written: {out_dir / 'fields.npz'}")
    for p in created_plots:
        print(f"written: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

