"""Rule-based suggested position and signal confidence.

- long:  bullish, confidence >= 0.55 and price above its 20-day average.
- short: bearish, confidence >= 0.55 and price below its 20-day average. Retail
         shorting is restricted in HK and Korea, so this means "avoid / hedge"
         (a `note` says so); the levels show where a hedge would be placed.
- none:  otherwise (entry/stop/target/risk_reward null), and always when the
         model's predicted return points the other way from the sentiment.
Entry = current price; stop just past the nearest support (long) or resistance
(short); target = the nearest level on the other side.
risk_reward = reward / risk.

Confidence (0-1) blends the strength of the sentiment shift (0.4, see
analytics/signal.py), post count (0.2), social/news agreement (0.2) and whether
the model's prediction points the same way (0.2). With no direction (sign 0) the
two agreement terms score 0.5.
"""

from __future__ import annotations

import math
from typing import Any

MIN_CONFIDENCE = 0.55
STOP_BUFFER = 0.005  # stop placed 0.5% beyond the level
AVOID_NOTE = "avoid / hedge: retail short selling is restricted in Hong Kong and Korea"


def _sign(x: float | None, dead: float = 0.0) -> int:
    if x is None or (isinstance(x, float) and math.isnan(x)) or abs(x) <= dead:
        return 0
    return 1 if x > 0 else -1


def confidence(direction_sign: int, strength: float, n_posts: int, news_sentiment: float | None,
               predicted_return: float | None) -> float:
    """direction_sign: +1 bullish, -1 bearish, 0 neutral; strength in [0, 1]."""
    s = _sign(float(direction_sign))
    strength = min(1.0, max(0.0, strength)) if s else 0.0
    volume = min(1.0, math.log1p(max(n_posts, 0)) / math.log1p(500))
    n, m = _sign(news_sentiment, 0.05), _sign(predicted_return)
    agree = 0.5 if (s == 0 or n == 0) else (1.0 if s == n else 0.0)
    model = 0.5 if (s == 0 or m == 0) else (1.0 if s == m else 0.0)
    return round(0.4 * strength + 0.2 * volume + 0.2 * agree + 0.2 * model, 3)


def _r(x: float | None, ref: float) -> float | None:
    if x is None:
        return None
    digits = 0 if ref >= 1000 else 2 if ref >= 1 else 4
    return round(float(x), digits)


def suggest_position(direction: str, conf: float, current: float, sma20: float | None,
                     support: list[float | None], resistance: list[float | None],
                     predicted_return: float | None = None) -> dict[str, Any]:
    none = {"side": "none", "entry": None, "stop": None, "target": None, "risk_reward": None}
    want = {"bullish": 1, "bearish": -1}.get(direction, 0)
    if want and _sign(predicted_return) == -want:
        return none  # sentiment and model disagree
    s1 = support[0] if support else None
    r1 = resistance[0] if resistance else None
    if sma20 is None or conf < MIN_CONFIDENCE or s1 is None or r1 is None:
        return none
    if direction == "bullish" and current > sma20:
        entry, stop, target = current, s1 * (1 - STOP_BUFFER), r1
        risk, reward = entry - stop, target - entry
        side, extra = "long", {}
    elif direction == "bearish" and current < sma20:
        entry, stop, target = current, r1 * (1 + STOP_BUFFER), s1
        risk, reward = stop - entry, entry - target
        side, extra = "short", {"note": AVOID_NOTE}
    else:
        return none
    if risk <= 0 or reward <= 0:
        return none
    return {"side": side, "entry": _r(entry, current), "stop": _r(stop, current), "target": _r(target, current),
            "risk_reward": round(reward / risk, 2), **extra}
