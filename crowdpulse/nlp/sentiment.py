"""Sentiment scores in [-1, 1] for posts and news, plus post weights.

Base model: cardiffnlp/twitter-xlm-roberta-base-sentiment (multilingual, built for
short social text). Score = P(positive) - P(negative), batches of 32 on CPU,
truncated to 256 tokens. Korean and Cantonese/Chinese finance slang lexicons in
nlp/lexicons/ adjust the score: when a term matches, the result is a 50/50 blend
of the model score and the lexicon score. Raw model scores are cached by id + text
hash in data/cache/sentiment.jsonl and the lexicon is applied on read, so reruns
only score new posts and lexicon edits need no re-scoring.

Optional LLM mode (--llm-sentiment, off by default): Claude scores 50 posts per
request, which handles sarcasm better. Needs the `anthropic` package and an
ANTHROPIC_API_KEY; it is never used unless asked for.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
FALLBACK_TOKENIZER = "FacebookAI/xlm-roberta-base"
LLM_MODEL = "claude-opus-5-5"
BATCH_SIZE = 32
MAX_TOKENS = 256
CHUNK_SIZE = 1000  # texts scored (and saved to the cache) per progress step
LLM_BATCH = 50
# When a lexicon term matches, the final score is a blend: (1 - w) * model + w * lexicon.
# The base model is a general tweet model; explicit market words are strong evidence.
LEXICON_WEIGHT = 0.5
BULLISH_THRESHOLD = 0.15
LEXICON_DIR = Path(__file__).parent / "lexicons"

PredictFn = Callable[[Sequence[str]], list[float]]


# ------------------------------------------------------------------- labels, weights


def label_for(score: float | None) -> str | None:
    """Contract rule: > 0.15 bullish, < -0.15 bearish, else neutral."""
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return None
    if score > BULLISH_THRESHOLD:
        return "bullish"
    if score < -BULLISH_THRESHOLD:
        return "bearish"
    return "neutral"


def post_weight(likes: Any, reposts: Any, views: Any, followers: Any) -> float:
    """1 + log(1 + likes + 2*reposts + views/100) + log(1 + author_followers). Missing -> 0."""
    def num(x: Any) -> float:
        try:
            v = float(x)
        except (TypeError, ValueError):
            return 0.0
        return 0.0 if math.isnan(v) or v < 0 else v

    return 1.0 + math.log1p(num(likes) + 2 * num(reposts) + num(views) / 100) + math.log1p(num(followers))


def weighted_mean(scores: Sequence[float], weights: Sequence[float]) -> float | None:
    s = np.asarray(scores, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = ~np.isnan(s) & ~np.isnan(w)
    if not mask.any() or w[mask].sum() <= 0:
        return None
    return float((s[mask] * w[mask]).sum() / w[mask].sum())


# --------------------------------------------------------------------------- lexicon


@dataclass(frozen=True)
class Lexicon:
    terms: dict[str, float]

    @classmethod
    def load(cls, directory: Path = LEXICON_DIR) -> "Lexicon":
        terms: dict[str, float] = {}
        for path in sorted(directory.glob("*.tsv")):
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                term, _, weight = line.partition("\t")
                if term and weight:
                    terms[term.strip()] = float(weight)
        return cls(terms)

    def score(self, text: str) -> float:
        """Sum of weights of terms present (each counted once), clipped to [-1, 1]."""
        total = sum(w for t, w in self.terms.items() if t in text)
        return max(-1.0, min(1.0, total))

    def hits(self, text: str) -> bool:
        return any(t in text for t in self.terms)

    def nudge(self, model_score: float, text: str, weight: float = LEXICON_WEIGHT) -> float:
        """Blend the model score with the lexicon score; unchanged when no term matches."""
        if not self.hits(text):
            return model_score
        blended = (1 - weight) * model_score + weight * self.score(text)
        return max(-1.0, min(1.0, blended))


# ---------------------------------------------------------------------- model scoring


def make_transformer_predictor(model_name: str = MODEL_NAME, batch_size: int = BATCH_SIZE,
                               max_tokens: int = MAX_TOKENS) -> PredictFn:
    """Load the HF model once; return texts -> P(pos) - P(neg)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except (ValueError, ImportError, OSError) as exc:
        # The cardiffnlp repo ships only sentencepiece.bpe.model, which needs the
        # `sentencepiece` package. It is fine-tuned from xlm-roberta-base and uses the
        # identical vocabulary, whose repo also ships a ready tokenizer.json.
        logger.warning("Tokenizer for %s failed (%s); using %s tokenizer (same vocabulary). "
                       "Installing `sentencepiece` avoids this fallback.", model_name, exc, FALLBACK_TOKENIZER)
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_TOKENIZER)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    labels = {v.lower(): int(k) for k, v in model.config.id2label.items()}
    pos, neg = labels.get("positive", 2), labels.get("negative", 0)

    def predict(texts: Sequence[str]) -> list[float]:
        # Batch texts of similar length together so short posts are not padded to
        # the longest one in a random batch (most posts are short); restore order after.
        order = sorted(range(len(texts)), key=lambda j: len(texts[j]))
        out: list[float] = [0.0] * len(texts)
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            batch = [texts[j] if texts[j] else " " for j in idx]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=max_tokens, return_tensors="pt")
            with torch.no_grad():
                probs = torch.softmax(model(**enc).logits, dim=-1).numpy()
            for j, s in zip(idx, (probs[:, pos] - probs[:, neg]).astype(float).tolist()):
                out[j] = s
        return out

    return predict


_LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"i": {"type": "integer"}, "score": {"type": "number"}},
                "required": ["i", "score"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scores"],
    "additionalProperties": False,
}

_LLM_SYSTEM = (
    "You score retail-investor social posts about one stock for market sentiment. "
    "Posts are Korean, Cantonese or Chinese and use slang and sarcasm. For each post return a "
    "score from -1 (clearly bearish on the stock) to 1 (clearly bullish); 0 for neutral, off-topic "
    "or unclear. Judge the author's view of the stock price, not their mood about other things."
)


def make_llm_predictor(client: Any = None, model: str = LLM_MODEL, batch: int = LLM_BATCH) -> PredictFn:
    """Claude scores `batch` posts per request with structured JSON output."""
    if client is None:
        import anthropic  # lazy: only needed with --llm-sentiment

        client = anthropic.Anthropic()

    def predict(texts: Sequence[str]) -> list[float]:
        out = [float("nan")] * len(texts)
        for start in range(0, len(texts), batch):
            chunk = texts[start:start + batch]
            listing = "\n".join(f"[{i}] {t[:600]}" for i, t in enumerate(chunk))
            resp = client.beta.messages.create(
                model=model,
                max_tokens=16000,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=_LLM_SYSTEM,
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": _LLM_SCHEMA}},
                messages=[{"role": "user", "content": f"Score these {len(chunk)} posts:\n{listing}"}],
            )
            if resp.stop_reason == "refusal":
                logger.warning("LLM declined a sentiment batch; those posts fall back to the base model")
                continue
            text = next((b.text for b in resp.content if b.type == "text"), "")
            try:
                for item in json.loads(text).get("scores", []):
                    i = int(item["i"])
                    if 0 <= i < len(chunk):
                        out[start + i] = max(-1.0, min(1.0, float(item["score"])))
            except (ValueError, KeyError, TypeError) as exc:
                logger.warning("Could not parse LLM sentiment batch: %s", exc)
        return out

    return predict


# ------------------------------------------------------------------------- the scorer


def _text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class SentimentScorer:
    """Scores texts with cache; base model + lexicon nudge, or LLM with base-model fallback.

    The cache stores the raw model score (flag ``"n": 1``) and the lexicon nudge is
    applied when reading, so editing the lexicons never requires re-scoring. LLM
    scores are stored as final (no nudge).
    """

    def __init__(self, cache_path: Path | None = None, *, predict: PredictFn | None = None,
                 llm_predict: PredictFn | None = None, lexicon: Lexicon | None = None,
                 model_name: str = MODEL_NAME, chunk_size: int = CHUNK_SIZE):
        self.cache_path = cache_path
        self.chunk_size = max(1, chunk_size)
        self._predict = predict
        self.llm_predict = llm_predict
        self.lexicon = lexicon or Lexicon.load()
        # "raw:" keys hold un-nudged model scores; older "model:" entries (nudged) are ignored.
        self.mode = f"llm:{LLM_MODEL}" if llm_predict else f"raw:{model_name}"
        self.model_name = model_name
        self.cache: dict[str, tuple[float, bool]] = {}  # key -> (score, apply lexicon nudge)
        if cache_path and cache_path.is_file():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                    self.cache[rec["k"]] = (float(rec["s"]), bool(rec.get("n", 0)))
                except (ValueError, KeyError, TypeError):
                    continue

    @property
    def predict(self) -> PredictFn:
        if self._predict is None:
            logger.info("Loading sentiment model %s (first run downloads ~1 GB)", self.model_name)
            self._predict = make_transformer_predictor(self.model_name)
        return self._predict

    def _key(self, item_id: str, text: str) -> str:
        return f"{self.mode}|{item_id}|{_text_hash(text)}"

    def score(self, ids: Sequence[str], texts: Sequence[str]) -> list[float]:
        keys = [self._key(i, t) for i, t in zip(ids, texts)]
        todo = [j for j, k in enumerate(keys) if k not in self.cache]
        if todo:
            logger.info("Scoring %d new texts (%d cached)", len(todo), len(keys) - len(todo))
            started = time.monotonic()
            # Work in chunks and save each one, so an interrupted run keeps its progress
            # and the log shows how far along it is.
            for c in range(0, len(todo), self.chunk_size):
                chunk = todo[c:c + self.chunk_size]
                self._score_chunk([keys[j] for j in chunk], [texts[j] for j in chunk])
                done = c + len(chunk)
                rate = done / max(time.monotonic() - started, 1e-9)
                logger.info("Sentiment: %d/%d scored (%.1f/s, ~%.0f min left)", done, len(todo), rate,
                            (len(todo) - done) / rate / 60)
        out = []
        for k, text in zip(keys, texts):
            s, nudge = self.cache[k]
            out.append(round(self.lexicon.nudge(s, text), 4) if nudge else s)
        return out

    def _score_chunk(self, keys: list[str], texts: list[str]) -> None:
        scores: list[float] = [float("nan")] * len(texts)
        if self.llm_predict is not None:
            scores = self.llm_predict(texts)
        nudge = [False] * len(texts)
        missing = [n for n, s in enumerate(scores) if s is None or math.isnan(s)]
        if missing:
            base = self.predict([texts[n] for n in missing])
            for n, s in zip(missing, base):
                scores[n], nudge[n] = s, True
        new = {}
        for k, s, n in zip(keys, scores, nudge):
            self.cache[k] = new[k] = (round(float(s), 4), n)
        self._append_cache(new)

    def _append_cache(self, new: dict[str, tuple[float, bool]]) -> None:
        if not self.cache_path or not new:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as fh:
            for k, (s, n) in new.items():
                fh.write(json.dumps({"k": k, "s": s, "n": int(n)}, ensure_ascii=False) + "\n")


def default_scorer(data_dir: Path, *, llm: bool = False) -> SentimentScorer:
    return SentimentScorer(data_dir / "cache" / "sentiment.jsonl",
                           llm_predict=make_llm_predictor() if llm else None)


def score_posts(posts: pd.DataFrame, scorer: SentimentScorer) -> pd.DataFrame:
    """Fill sentiment_score, sentiment_label and weight on a posts DataFrame."""
    out = posts.copy()
    if out.empty:
        return out
    out["sentiment_score"] = scorer.score(out["post_id"].astype(str).tolist(), out["text"].astype(str).tolist())
    out["sentiment_label"] = out["sentiment_score"].map(label_for)
    out["weight"] = [post_weight(*r) for r in out[["likes", "reposts", "views", "author_followers"]]
                     .itertuples(index=False, name=None)]
    return out


def score_news(news: pd.DataFrame, scorer: SentimentScorer) -> pd.DataFrame:
    """Fill sentiment_score on news from title + summary."""
    out = news.copy()
    if out.empty:
        return out
    texts = [" ".join(str(x) for x in (r.title, r.summary) if isinstance(x, str) and x)
             for r in out.itertuples(index=False)]
    ids = ["news:" + _text_hash(str(u)) for u in out["url"]]
    out["sentiment_score"] = scorer.score(ids, texts)
    return out
