"""Accuracy of the sentiment scorer on 20 hand-labelled posts (10 Korean, 10 Chinese).

    python scripts/eval_sentiment.py

Prints 3-class accuracy (bullish / bearish / neutral) for the base model alone and
with the slang lexicon nudge, overall and per language. Downloads the model on
first use; does not touch the score cache.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crowdpulse.nlp.sentiment import Lexicon, label_for, make_transformer_predictor  # noqa: E402

SAMPLES = ROOT / "tests" / "backend" / "samples" / "labelled_posts.json"


def evaluate(posts: list[dict], raw_scores: list[float], lexicon: Lexicon) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for mode in ("model", "model+lexicon"):
        preds = [label_for(s if mode == "model" else lexicon.nudge(s, p["text"])) for s, p in zip(raw_scores, posts)]
        for lang in ("all", "ko", "zh"):
            idx = [i for i, p in enumerate(posts) if lang == "all" or p["lang"] == lang]
            hits = sum(preds[i] == posts[i]["label"] for i in idx)
            out.setdefault(mode, {})[lang] = hits / len(idx) if idx else float("nan")
    return out


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    posts = json.loads(SAMPLES.read_text(encoding="utf-8"))
    lexicon = Lexicon.load()
    raw = make_transformer_predictor()([p["text"] for p in posts])
    for p, s in zip(posts, raw):
        nudged = lexicon.nudge(s, p["text"])
        print(f"{p['lang']} {p['label']:>7} | model {s:+.2f} -> {label_for(s):>7} | "
              f"+lexicon {nudged:+.2f} -> {label_for(nudged):>7} | {p['text'][:40]}")
    res = evaluate(posts, raw, lexicon)
    print()
    for mode, accs in res.items():
        print(f"{mode:>14}: all {accs['all']:.0%}   korean {accs['ko']:.0%}   chinese {accs['zh']:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
