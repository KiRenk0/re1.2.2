#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从当前工程求解器输出（runs/<run_dir>/summary.json + fields.npz）画“根弦沿弦线”温度/热流曲线。

用法：
  python scripts/plot_root_chord_temperature_from_run_rem.py --run_dir runs/rem_demo

要求：
- sampling.mode 必须是 root_windward_chord_line（ny=1，沿弦向 1D）
- fields.npz 里需要有 Tw_w（以及可选 q_w/Tw_l/q_l）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _ensure_import_path() -> None:
    here = Path(__file__).resolve()
    repo_root = here.parents[1]
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot root chord 1D profiles (Tw/q) from run_dir outputs.")
    ap.add_argument("--run_dir", default="runs/rem_demo")
    ap.add_argument("--out_dir", default=None, help="默认写到 run_dir 里")
    ap.add_argument("--also_q", action="store_true", help="也画热流 q（若 fields.npz 里有）")
    ap.add_argument("--side", default="windward", choices=["windward", "leeward"])
    ap.add_argument("--x_over_c_min", type=float, default=0.01, help="只画 x/c >= 此值（默认 0.01）")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    summary_path = run_dir / "summary.json"
    fields_path = run_dir / "fields.npz"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing: {summary_path}")
    if not fields_path.exists():
        raise FileNotFoundError(f"missing: {fields_path}")

    summary = _load_json(summary_path)
    fields = np.load(fields_path, allow_pickle=True)

    sampling_path = Path(summary["resolved_paths"]["sampling_path"])
    vehicle_path = Path(summary["resolved_paths"]["vehicle_path"])
    sampling = _load_yaml(sampling_path)["canonical_sampling_spec"]
    vehicle = _load_yaml(vehicle_path)["vehicle_spec"]

    mode = str(sampling.get("mode", "")).strip()
    if mode != "root_windward_chord_line":
        raise ValueError(f"sampling.mode={mode!r}，不是 root_windward_chord_line；该脚本用于 1D 根弦线出图。")

    nx = int(sampling["x_over_c"]["n"])
    if int(getattr(sampling.get("y_over_b", 0.0), "__len__", lambda: 0)()) != 0:
        # y_over_b 在这个 mode 下通常是标量
        pass

    xc = np.linspace(float(sampling["x_over_c"]["start"]), float(sampling["x_over_c"]["end"]), nx)
    c_root = float(vehicle["planform"]["c_root_m"])
    x_m = xc * c_root
    mask = xc >= max(float(args.x_over_c_min), 0.0)

    key_tw = "Tw_w" if args.side == "windward" else "Tw_l"
    if key_tw not in fields:
        raise KeyError(f"{fields_path} 中没有 {key_tw}")
    Tw = np.asarray(fields[key_tw], dtype=float).reshape(-1)
    if Tw.size != nx:
        raise ValueError(f"{key_tw}.size={Tw.size} != nx={nx}")

    key_q = "q_w" if args.side == "windward" else "q_l"
    q = None
    if bool(args.also_q) and key_q in fields:
        q = np.asarray(fields[key_q], dtype=float).reshape(-1)
        if q.size != nx:
            q = None

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("缺少 matplotlib。请先执行：pip install -r requirements.txt") from e

    out_dir = run_dir if args.out_dir is None else Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_mach = summary.get("inputs", {}).get("mach", None)
    meta_alpha = summary.get("inputs", {}).get("alpha_deg", None)
    h_m = summary.get("freestream", {}).get("h_m", summary.get("inputs", {}).get("h_m_override", None))
    h_km = (None if h_m is None else float(h_m) / 1000.0)
    h_text = ("h=?" if h_km is None else f"h={h_km:.2f} km")

    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=170)
    ax.plot(x_m[mask], Tw[mask], linewidth=1.6)
    ax.set_xlabel("x / m")
    ax.set_ylabel("T / K")
    ax.set_title(f"Tw root chord ({args.side}) | {h_text}, M={meta_mach}, alpha={meta_alpha} deg")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    fig.tight_layout()

    out_tw = out_dir / f"Tw_root_chord_{args.side}.png"
    fig.savefig(out_tw)
    print(out_tw)

    if q is not None:
        fig2, ax2 = plt.subplots(figsize=(7.6, 4.8), dpi=170)
        ax2.plot(x_m[mask], q[mask], linewidth=1.6)
        ax2.set_xlabel("x / m")
        ax2.set_ylabel("q / W m$^{-2}$")
        ax2.set_title(f"q root chord ({args.side}) | {h_text}, M={meta_mach}, alpha={meta_alpha} deg")
        ax2.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        fig2.tight_layout()
        out_q = out_dir / f"q_root_chord_{args.side}.png"
        fig2.savefig(out_q)
        print(out_q)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

