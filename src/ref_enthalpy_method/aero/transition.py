"""Boundary-layer transition criterion.

Implements eq. (2.46):
    (Re_tri)_e = 10^(5.37 + 0.2326 Ma_e - 0.004015 Ma_e^2)
"""

from __future__ import annotations

import math


def transition_reynolds(*, ma_e: float) -> float:
    mae = float(ma_e)
    expo = 5.37 + 0.2326 * mae - 0.004015 * (mae**2)
    return 10.0**expo


def transition_weight(
    *,
    enable: bool = True,
    re_measure: float,
    re_tri: float,
    weighting: str = "logistic",
    width_decades: float = 0.25,
    x_over_c: float | None = None,
    transition_x_over_c: float | None = None,
) -> float:
    """Return w in [0,1] (0=laminar, 1=turbulent).

    This is an engineering helper (baseline-compatible), not from the doc directly.
    - weighting="step": hard switch at Re >= Re_tri
    - weighting="logistic": smooth blend in log10(Re/Re_tri) over `width_decades`
    - if transition_x_over_c is provided, forbid transition before that x/c.
    """

    if not bool(enable):
        return 0.0

    re_tri = float(re_tri)
    if not math.isfinite(re_tri) or re_tri <= 0:
        return 0.0

    re_measure = float(re_measure)
    if not math.isfinite(re_measure) or re_measure <= 0:
        return 0.0

    if transition_x_over_c is not None and x_over_c is not None:
        try:
            if float(x_over_c) < float(transition_x_over_c):
                return 0.0
        except Exception:
            pass

    mode = str(weighting).strip().lower()
    if mode in {"step", "hard"}:
        return 1.0 if re_measure >= re_tri else 0.0

    # IMPORTANT: For doc reproduction we treat the criterion as a threshold:
    # if the measured Reynolds number does not reach Re_tri, keep fully-laminar.
    # The logistic blend is only used after crossing the threshold, to avoid
    # introducing a "pre-transition" turbulent contribution that can lift the
    # aft-chord temperatures even when Re < Re_tri everywhere.
    if re_measure < re_tri:
        return 0.0

    width = max(float(width_decades), 1e-6)
    z = math.log10(re_measure / re_tri) / width
    z = max(min(z, 60.0), -60.0)
    return float(1.0 / (1.0 + math.exp(-z)))

