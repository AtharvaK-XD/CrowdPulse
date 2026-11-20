"""Social-sentiment direction relative to the stock's own recent baseline.

Raw scores carry a platform-wide tone: on the first live run every Korean stock's
Naver posts averaged about -0.27, because forum posts are often angry or rude
whatever the stock does. The absolute level therefore says more about the forum
than about the stock, and labelling it directly made every KRX stock "bearish".

The direction is instead whether recent posts are clearly more positive or more
negative than the stock's own earlier posts in the window:

- recent   = posts assigned to the latest closed bar or later (since the close
             before last, including after-close posts that belong to the next bar);
- baseline = earlier posts in the window;
- shift    = weighted mean(recent) - weighted mean(baseline);
- z        = shift / standard error (weighted means, Kish effective sample size).

bullish if z >= MIN_Z and shift >= MIN_SHIFT, bearish if the mirror image,
neutral otherwise, including when either side has fewer than MIN_POSTS posts.
strength = min(1, |z| / Z_FULL) for a non-neutral direction, else 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_POSTS = 5
MIN_Z = 2.0
MIN_SHIFT = 0.03
Z_FULL = 4.0


@dataclass(frozen=True)
class SentimentShift:
    recent: float | None
    baseline: float | None
    shift: float | None
    z: float | None
    n_recent: int
    n_baseline: int

    @property
    def direction(self) -> str:
        if self.z is None or self.shift is None:
            return "neutral"
        if self.z >= MIN_Z and self.shift >= MIN_SHIFT:
            return "bullish"
        if self.z <= -MIN_Z and self.shift <= -MIN_SHIFT:
            return "bearish"
        return "neutral"

    @property
    def sign(self) -> int:
        return {"bullish": 1, "bearish": -1}.get(self.direction, 0)

    @property
    def strength(self) -> float:
        if self.sign == 0 or self.z is None:
            return 0.0
        return min(1.0, abs(self.z) / Z_FULL)


def _weighted_stats(scores: np.ndarray, weights: np.ndarray) -> tuple[float, float, float] | None:
    """Weighted mean, weighted variance and Kish effective sample size."""
    mask = ~np.isnan(scores) & ~np.isnan(weights) & (weights > 0)
    s, w = scores[mask], weights[mask]
    if len(s) == 0:
        return None
    mean = float((s * w).sum() / w.sum())
    var = float((w * (s - mean) ** 2).sum() / w.sum())
    n_eff = float(w.sum() ** 2 / (w ** 2).sum())
    return mean, var, n_eff


def sentiment_shift(posts: pd.DataFrame, last_bar: str) -> SentimentShift:
    """Compare posts on/after ``last_bar`` (ISO date) with earlier posts of one ticker."""
    if posts.empty or "bar_date" not in posts:
        return SentimentShift(None, None, None, None, 0, 0)
    bar = posts["bar_date"].astype(str)
    scores = pd.to_numeric(posts["sentiment_score"], errors="coerce").to_numpy(dtype=float)
    weights = pd.to_numeric(posts["weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    is_recent = (bar >= last_bar).to_numpy()
    n_r, n_b = int(is_recent.sum()), int((~is_recent).sum())
    r = _weighted_stats(scores[is_recent], weights[is_recent])
    b = _weighted_stats(scores[~is_recent], weights[~is_recent])
    recent = r[0] if r else None
    baseline = b[0] if b else None
    if r is None or b is None or n_r < MIN_POSTS or n_b < MIN_POSTS:
        return SentimentShift(recent, baseline, None, None, n_r, n_b)
    shift = r[0] - b[0]
    se = math.sqrt(r[1] / r[2] + b[1] / b[2])
    z = shift / se if se > 0 else (0.0 if shift == 0 else math.copysign(math.inf, shift))
    return SentimentShift(recent, baseline, shift, z, n_r, n_b)
