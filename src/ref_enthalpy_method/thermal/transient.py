"""Transient wall-energy balance (Doc Eq 2.57) time marching utilities."""

from __future__ import annotations

import numpy as np


def require_transient_material(cfg: dict) -> None:
    missing = []
    for k in ("rho_wall_kg_m3", "c_wall_J_per_kgK", "delta_wall_m"):
        if cfg.get(k, None) is None:
            missing.append(k)
    if missing:
        raise ValueError(
            "tw_model.type='transient_balance' requires tw_model.transient fields: "
            f"{missing}. Example:\n"
            "tw_model:\n"
            "  type: transient_balance\n"
            "  sigma: 5.76e-8\n"
            "  transient:\n"
            "    rho_wall_kg_m3: 2700\n"
            "    c_wall_J_per_kgK: 900\n"
            "    delta_wall_m: 0.002\n"
            "    dt_s: 0.01\n"
            "    t_end_s: 10.0\n"
            "    Tw_init_K: 300.0\n"
        )


def march_explicit_balance(
    *,
    Tw0: np.ndarray,
    dt_s: float,
    n_steps: int,
    cap_J_per_m2K: float,
    emissivity: float,
    sigma_W_m2_K4: float,
    Tw_min_K: float,
    Tw_max_K: float,
    eval_q_a,
) -> tuple[np.ndarray, np.ndarray]:
    """Explicit Euler marching for:

    cap * dTw/dt = q_a(Tw) - eps*sigma*Tw^4
    """

    Tw0 = np.asarray(Tw0, dtype=float).reshape(-1)
    n = int(Tw0.size)
    dt = float(dt_s)
    cap = float(cap_J_per_m2K)
    if cap <= 0:
        raise ValueError("Invalid wall heat capacity per area: rho*c*delta must be > 0")
    if dt <= 0:
        raise ValueError("Invalid transient time settings: dt_s must be > 0")
    n_steps = int(n_steps)
    if n_steps < 0:
        raise ValueError("Invalid transient time settings: n_steps must be >= 0")

    Tw_time = np.full((n_steps + 1, n), np.nan, dtype=float)
    q_time = np.full((n_steps + 1, n), np.nan, dtype=float)
    Tw_time[0, :] = np.clip(Tw0, float(Tw_min_K), float(Tw_max_K))

    for k in range(n_steps):
        Tw_k = Tw_time[k, :].copy()
        q_k = np.asarray(eval_q_a(Tw_k), dtype=float).reshape(-1)
        if q_k.size != n:
            raise ValueError(f"eval_q_a returned shape {q_k.shape}, expected ({n},)")
        q_time[k, :] = q_k

        Tw_next = Tw_k.copy()
        for i in range(n):
            Tw_i = float(Tw_k[i])
            q_i = float(q_k[i])
            if not np.isfinite(Tw_i) or not np.isfinite(q_i):
                continue
            q_r = float(emissivity) * float(sigma_W_m2_K4) * (Tw_i**4)
            dTdt = (q_i - q_r) / cap
            Tw_next[i] = np.clip(Tw_i + dt * dTdt, float(Tw_min_K), float(Tw_max_K))

        Tw_time[k + 1, :] = Tw_next

    q_time[-1, :] = np.asarray(eval_q_a(Tw_time[-1, :]), dtype=float).reshape(-1)
    return Tw_time, q_time


def march_explicit_balance_final(
    *,
    Tw0: np.ndarray,
    dt_s: float,
    n_steps: int,
    cap_J_per_m2K: float,
    emissivity: float,
    sigma_W_m2_K4: float,
    Tw_min_K: float,
    Tw_max_K: float,
    eval_q_a,
) -> tuple[np.ndarray, np.ndarray]:
    """Memory-light variant returning only final (Tw, q) instead of full histories."""

    Tw0 = np.asarray(Tw0, dtype=float).reshape(-1)
    n = int(Tw0.size)
    dt = float(dt_s)
    cap = float(cap_J_per_m2K)
    if cap <= 0:
        raise ValueError("Invalid wall heat capacity per area: rho*c*delta must be > 0")
    if dt <= 0:
        raise ValueError("Invalid transient time settings: dt_s must be > 0")
    n_steps = int(n_steps)
    if n_steps < 0:
        raise ValueError("Invalid transient time settings: n_steps must be >= 0")

    Tw = np.clip(Tw0, float(Tw_min_K), float(Tw_max_K))
    q = np.full((n,), np.nan, dtype=float)

    for _k in range(n_steps):
        q[:] = np.asarray(eval_q_a(Tw), dtype=float).reshape(-1)
        for i in range(n):
            Tw_i = float(Tw[i])
            q_i = float(q[i])
            if not np.isfinite(Tw_i) or not np.isfinite(q_i):
                continue
            q_r = float(emissivity) * float(sigma_W_m2_K4) * (Tw_i**4)
            dTdt = (q_i - q_r) / cap
            Tw[i] = np.clip(Tw_i + dt * dTdt, float(Tw_min_K), float(Tw_max_K))

    q[:] = np.asarray(eval_q_a(Tw), dtype=float).reshape(-1)
    return np.asarray(Tw, dtype=float), np.asarray(q, dtype=float)

