from __future__ import annotations

import math

import pandas as pd
import pytest

from crowdpulse.nlp.sentiment import (
    Lexicon, SentimentScorer, label_for, make_llm_predictor, post_weight, score_news, score_posts, weighted_mean,
)


def test_label_thresholds():
    assert label_for(0.16) == "bullish"
    assert label_for(0.15) == "neutral"
    assert label_for(-0.15) == "neutral"
    assert label_for(-0.16) == "bearish"
    assert label_for(None) is None


def test_post_weight_formula():
    w = post_weight(10, 2, 500, 99)
    assert w == pytest.approx(1 + math.log(1 + 10 + 4 + 5) + math.log(1 + 99))
    assert post_weight(None, None, None, None) == 1.0
    assert post_weight(float("nan"), 0, 0, 0) == 1.0


def test_weighted_mean():
    assert weighted_mean([1.0, -1.0], [3.0, 1.0]) == pytest.approx(0.5)
    assert weighted_mean([], []) is None
    assert weighted_mean([float("nan")], [1.0]) is None


def test_lexicons_load_and_nudge():
    lex = Lexicon.load()
    for term in ("상한가", "떡상", "하한가", "개미털기", "入貨", "沽貨", "爆倉", "河豚年紀"):
        assert term in lex.terms
    assert lex.score("오늘 상한가 떡상") == 1.0          # clipped
    assert lex.score("騰訊爆倉") < 0
    assert lex.nudge(0.0, "중립 문장") == 0.0
    assert lex.nudge(0.9, "상한가") == pytest.approx(0.95)  # blend 50/50 with lexicon score 1.0
    assert lex.nudge(0.0, "하한가") == pytest.approx(-0.5)
    # A confidently wrong model score is pulled back by an explicit market word.
    assert lex.nudge(0.57, "내일도 100%하락이다") == pytest.approx(-0.015)


def test_cache_stores_raw_model_score_and_nudges_on_read(tmp_path):
    cache = tmp_path / "s.jsonl"
    s = SentimentScorer(cache, predict=CountingPredict(0.4), lexicon=Lexicon({"상한가": 1.0}))
    assert s.score(["a", "b"], ["상한가", "x"]) == [pytest.approx(0.7), 0.4]
    assert '"s": 0.4' in cache.read_text(encoding="utf-8").splitlines()[0]   # raw, not nudged
    # A lexicon change applies to cached scores without re-scoring.
    s2 = SentimentScorer(cache, predict=CountingPredict(-1.0), lexicon=Lexicon({"상한가": -1.0}))
    assert s2.score(["a"], ["상한가"]) == [pytest.approx(-0.3)]


class CountingPredict:
    def __init__(self, value=0.5):
        self.value = value
        self.seen: list[str] = []

    def __call__(self, texts):
        self.seen.extend(texts)
        return [self.value] * len(texts)


def test_scorer_caches_by_id_and_text(tmp_path):
    cache = tmp_path / "s.jsonl"
    pred = CountingPredict(0.2)
    s = SentimentScorer(cache, predict=pred, lexicon=Lexicon({}))
    assert s.score(["a", "b"], ["x", "y"]) == [0.2, 0.2]
    s2 = SentimentScorer(cache, predict=CountingPredict(0.9), lexicon=Lexicon({}))
    assert s2.score(["a", "b"], ["x", "y"]) == [0.2, 0.2]       # from disk cache
    assert s2.score(["a"], ["x edited"]) == [0.9]               # text changed -> rescored


def test_llm_mode_falls_back_to_model_for_missing(tmp_path):
    def llm(texts):
        return [0.7, float("nan")]

    s = SentimentScorer(None, predict=CountingPredict(-0.4), llm_predict=llm, lexicon=Lexicon({}))
    assert s.score(["a", "b"], ["x", "y"]) == [0.7, -0.4]


def test_llm_predictor_parses_structured_output():
    class Block:
        type = "text"
        text = '{"scores": [{"i": 0, "score": 0.8}, {"i": 1, "score": -2}]}'

    class Resp:
        stop_reason = "end_turn"
        content = [Block()]

    class Client:
        def __init__(self):
            self.beta = self
            self.messages = self
            self.kwargs = None

        def create(self, **kw):
            self.kwargs = kw
            return Resp()

    c = Client()
    out = make_llm_predictor(client=c)(["좋다", "나쁘다"])
    assert out == [0.8, -1.0]
    assert c.kwargs["model"] == "claude-opus-5-5"
    assert c.kwargs["output_config"]["format"]["type"] == "json_schema"


def test_llm_refusal_leaves_nan():
    class Resp:
        stop_reason = "refusal"
        content = []

    class Client:
        def __init__(self):
            self.beta = self
            self.messages = self

        def create(self, **kw):
            return Resp()

    out = make_llm_predictor(client=Client())(["x"])
    assert math.isnan(out[0])


def test_score_posts_and_news():
    s = SentimentScorer(None, predict=CountingPredict(0.5), lexicon=Lexicon({}))
    posts = pd.DataFrame({"post_id": ["p1", "p2"], "text": ["a", "b"], "likes": [1, 0], "reposts": [0, 0],
                          "views": [100, 0], "author_followers": [0, 0]})
    out = score_posts(posts, s)
    assert out["sentiment_score"].tolist() == [0.5, 0.5]
    assert out["sentiment_label"].tolist() == ["bullish", "bullish"]
    assert out["weight"].iloc[0] > out["weight"].iloc[1] == 1.0
    news = pd.DataFrame({"title": ["t"], "summary": [None], "url": ["https://x"]})
    assert score_news(news, s)["sentiment_score"].tolist() == [0.5]


def test_scorer_saves_each_chunk(tmp_path):
    cache = tmp_path / "s.jsonl"
    calls = []

    def pred(texts):
        calls.append(list(texts))
        return [0.1] * len(texts)

    s = SentimentScorer(cache, predict=pred, lexicon=Lexicon({}), chunk_size=2)
    assert s.score(["a", "b", "c"], ["x", "y", "z"]) == [0.1, 0.1, 0.1]
    assert calls == [["x", "y"], ["z"]]
    assert len(cache.read_text(encoding="utf-8").splitlines()) == 3
