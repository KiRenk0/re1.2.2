#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从当前工程求解器输出（runs/<run_dir>/summary.json + fields.npz）画半翼面温度场（Tw）。

用法（先跑一次 full_wing_surface_grid 的 run）：
  python scripts/plot_wing_surface_temperature_from_run_rem.py --run_dir runs/rem_demo_2d --side windward

要求：
- sampling.mode 必须是 full_wing_surface_grid（才能重建 (x/c,y/b) 网格）
- fields.npz 里必须有 Tw_w/Tw_l
  - 固定壁温也会有 Tw_*（常数）
  - radiative_equilibrium 会有 Tw_*（随位置变化）
  - transient_balance 默认只保存最终态 Tw_*（ny>1 时不保存全时序）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _ensure_import_path() -> None:
    """Ensure `import ref_enthalpy_method` works for src-layout."""
    here = Path(__file__).resolve()
    repo_root = here.parents[1]  # .../<repo>/scripts/xxx.py
    src_root = repo_root / "src"
    for p in (str(repo_root), str(src_root)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _load_json(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(p: Path) -> dict:
    _ensure_import_path()
    from ref_enthalpy_method.specs.loader import load_yaml  # noqa: WPS433

    return load_yaml(p)


def _triangulate_structured(ny: int, nx: int) -> np.ndarray:
    """把 (ny,nx) 结构网格拆成两三角/单元，用于 tricontourf。"""
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot half-wing surface temperature Tw(x,y) from run_dir outputs.")
    ap.add_argument("--run_dir", default="runs/rem_demo", help="runs 下的子目录（含 summary.json + fields.npz）")
    ap.add_argument("--side", default="windward", choices=["windward", "leeward"])
    ap.add_argument("--cmap", default="turbo")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--x_over_c_min", type=float, default=0.00, help="只画 x/c >= 此值（默认 0.01）")
    ap.add_argument("--out", default=None, help="输出图片路径；默认写到 run_dir 里")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    summary_path = run_dir / "summary.json"
    fields_path = run_dir / "fields.npz"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing: {summary_path}")
    if not fields_path.exists():
        raise FileNotFoundError(f"missing: {fields_path} (请先用 scripts/run_case_rem.py 生成)")

    summary = _load_json(summary_path)
    fields = np.load(fields_path, allow_pickle=True)

    key_tw = "Tw_w" if args.side == "windward" else "Tw_l"
    if key_tw not in fields:
        raise KeyError(
            f"{fields_path} 中没有 {key_tw}。"
            "需要先跑出 Tw_*（例如 tw_model.type=radiative_equilibrium 或固定壁温），并确保保存了 fields.npz。"
        )

    Tw_flat = np.asarray(fields[key_tw], dtype=float).reshape(-1)

    sampling_path = Path(summary["resolved_paths"]["sampling_path"])
    vehicle_path = Path(summary["resolved_paths"]["vehicle_path"])
    sampling = _load_yaml(sampling_path)["canonical_sampling_spec"]
    vehicle = _load_yaml(vehicle_path)["vehicle_spec"]

    mode = str(sampling.get("mode", "")).strip()
    if mode != "full_wing_surface_grid":
        raise ValueError(f"sampling.mode={mode!r}，不是 full_wing_surface_grid；该脚本用于画翼面 2D 温度场。")

    nx = int(sampling["x_over_c"]["n"])
    ny = int(sampling["y_over_b"]["n"])
    if Tw_flat.size != nx * ny:
        raise ValueError(f"{key_tw}.size={Tw_flat.size} != nx*ny={nx*ny}，请确认 sampling 与输出是否匹配。")

    Tw = Tw_flat.reshape((ny, nx))

    # 重建网格（单位：m）
    xc = np.linspace(float(sampling["x_over_c"]["start"]), float(sampling["x_over_c"]["end"]), nx)
    yb = np.linspace(float(sampling["y_over_b"]["start"]), float(sampling["y_over_b"]["end"]), ny)
    col_mask = xc >= max(float(args.x_over_c_min), 0.0)
    if not np.any(col_mask):
        col_mask = np.ones_like(xc, dtype=bool)
    xc = xc[col_mask]
    Tw = Tw[:, col_mask]
    nx = int(xc.size)

    b_half = float(vehicle["planform"]["b_half_m"])
    c_root = float(vehicle["planform"]["c_root_m"])
    c_tip = float(vehicle["planform"]["c_tip_m"])
    sweep_deg = float(vehicle["planform"]["sweep_le_deg"])
    chi = np.deg2rad(sweep_deg)

    X = np.zeros((ny, nx), dtype=float)
    Y = np.zeros((ny, nx), dtype=float)
    for j in range(ny):
        y = float(yb[j]) * b_half
        chord = c_root + (c_tip - c_root) * float(yb[j])
        x_le = y * np.tan(chi)
        Y[j, :] = y
        X[j, :] = x_le + xc * chord

    # 出图
    try:
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
    except Exception as e:  # pragma: no cover
        raise RuntimeError("缺少 matplotlib。请先执行：pip install -r requirements.txt") from e

    fig, ax = plt.subplots(figsize=(7.8, 4.6), dpi=170)
    x = X.reshape(-1)
    y = Y.reshape(-1)
    z = Tw.reshape(-1)
    tri = mtri.Triangulation(x, y, _triangulate_structured(ny, nx))

    finite = np.isfinite(z)
    if not np.any(finite):
        raise ValueError("Tw 全是 NaN/Inf，无法出图（请检查上游 run 是否正常）。")

    vmin = float(args.vmin) if args.vmin is not None else float(np.nanmin(z))
    vmax = float(args.vmax) if args.vmax is not None else float(np.nanmax(z))
    if not (np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin):
        vmin, vmax = float(np.nanmin(z[finite])), float(np.nanmax(z[finite]))
        if not (vmax > vmin):
            vmax = vmin + 1.0

    levels = np.linspace(vmin, vmax, 40)
    im = ax.tricontourf(tri, z, levels=levels, cmap=str(args.cmap))
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("T / K")

    # 平面外形轮廓（半翼）
    y0 = 0.0
    y1 = float(b_half)
    x_le_root = 0.0
    x_le_tip = y1 * np.tan(chi)
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
    ax.set_title(f"Tw surface ({args.side}) | {h_text}, M={meta_mach}, alpha={meta_alpha} deg")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m (half-span)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    fig.tight_layout()

    out = (run_dir / f"Tw_surface_{args.side}.png") if args.out is None else Path(args.out).resolve()
    fig.savefig(out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

