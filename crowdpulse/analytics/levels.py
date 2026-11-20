"""Support and resistance from ~120 days of daily bars.

1. Swing highs/lows: a bar whose high (low) is above (below) the 3 bars on each side.
2. Swings within 1.5% of each other merge into one level (mean price), ranked by touches.
3. Output the 2 nearest supports below and the 2 nearest resistances above the
   current price. If swings don't supply enough, classic pivot points from the last
   bar (P, S1, S2, R1, R2) fill the gaps, then the period low/high.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SWING_WINDOW = 3
MERGE_TOLERANCE = 0.015


@dataclass
class Level:
    price: float
    touches: int


@dataclass
class Levels:
    support: list[float | None]
    resistance: list[float | None]
    method: str
    detail: dict = field(default_factory=dict)


def swing_points(high: np.ndarray, low: np.ndarray, k: int = SWING_WINDOW) -> tuple[list[float], list[float]]:
    """(swing highs, swing lows): strictly above/below the k bars on each side."""
    highs, lows = [], []
    for i in range(k, len(high) - k):
        left_h, right_h = high[i - k:i], high[i + 1:i + k + 1]
        if high[i] > left_h.max() and high[i] > right_h.max():
            highs.append(float(high[i]))
        left_l, right_l = low[i - k:i], low[i + 1:i + k + 1]
        if low[i] < left_l.min() and low[i] < right_l.min():
            lows.append(float(low[i]))
    return highs, lows


def merge_levels(prices: list[float], tol: float = MERGE_TOLERANCE) -> list[Level]:
    """Greedy clustering of sorted prices: join while within tol of the cluster mean."""
    levels: list[Level] = []
    cluster: list[float] = []
    for p in sorted(prices):
        if cluster and abs(p - np.mean(cluster)) / np.mean(cluster) > tol:
            levels.append(Level(float(np.mean(cluster)), len(cluster)))
            cluster = []
        cluster.append(p)
    if cluster:
        levels.append(Level(float(np.mean(cluster)), len(cluster)))
    return sorted(levels, key=lambda lv: (-lv.touches, lv.price))


def pivot_points(high: float, low: float, close: float) -> dict[str, float]:
    p = (high + low + close) / 3
    return {"P": p, "R1": 2 * p - low, "S1": 2 * p - high, "R2": p + (high - low), "S2": p - (high - low)}


def _round(x: float | None, current: float) -> float | None:
    if x is None:
        return None
    digits = 0 if current >= 1000 else 2 if current >= 1 else 4
    return round(float(x), digits)


def compute_levels(bars: pd.DataFrame, current: float | None = None) -> Levels:
    """bars: one ticker's prices (date, high, low, close), oldest first."""
    bars = bars.sort_values("date")
    high, low, close = bars["high"].to_numpy(float), bars["low"].to_numpy(float), bars["close"].to_numpy(float)
    if len(close) == 0:
        return Levels([None, None], [None, None], "none")
    current = float(close[-1]) if current is None else float(current)

    sh, sl = swing_points(high, low)
    merged = merge_levels(sh + sl)
    below = sorted((lv for lv in merged if lv.price < current), key=lambda lv: current - lv.price)
    above = sorted((lv for lv in merged if lv.price > current), key=lambda lv: lv.price - current)
    support = [lv.price for lv in below[:2]]
    resistance = [lv.price for lv in above[:2]]
    method = "swing"

    if len(support) < 2 or len(resistance) < 2:
        pv = pivot_points(high[-1], low[-1], close[-1])
        method = "swing+pivot" if (support or resistance) else "pivot"
        for v in sorted((pv["S1"], pv["S2"], pv["P"]), reverse=True):
            if len(support) < 2 and v < current and all(abs(v - s) / s > MERGE_TOLERANCE for s in support):
                support.append(v)
        for v in sorted((pv["R1"], pv["R2"], pv["P"])):
            if len(resistance) < 2 and v > current and all(abs(v - r) / r > MERGE_TOLERANCE for r in resistance):
                resistance.append(v)
        support.sort(reverse=True)
        resistance.sort()
    if len(support) < 2 and low.min() < current and all(abs(low.min() - s) / s > MERGE_TOLERANCE for s in support):
        support.append(float(low.min()))
    if len(resistance) < 2 and high.max() > current and all(abs(high.max() - r) / r > MERGE_TOLERANCE for r in resistance):
        resistance.append(float(high.max()))

    support = (support + [None, None])[:2]
    resistance = (resistance + [None, None])[:2]
    return Levels(
        support=[_round(s, current) for s in support],
        resistance=[_round(r, current) for r in resistance],
        method=method,
        detail={"swing_highs": len(sh), "swing_lows": len(sl), "levels": len(merged)},
    )
