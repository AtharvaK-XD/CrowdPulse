# CrowdPulse — Shared Data Contract

## 1. Repo Layout and Owners
```text
crowdpulse/                  repo root
├── README.md                YOU
├── config.yaml              YOU  tickers, sources, dates, model settings
├── requirements.txt         YOU
├── .env.example             YOU  APIFY_TOKEN=, EXA_API_KEY=
├── run.py                   YOU  pipeline -> report (wired last)
├── CLAUDE.md                CLAUDE CODE
├── crowdpulse/              CLAUDE CODE  backend package
├── scripts/                 CLAUDE CODE
├── tests/backend/           CLAUDE CODE
├── AGENTS.md                ANTIGRAVITY
├── report/                  ANTIGRAVITY
├── fixtures/                ANTIGRAVITY
├── tests/report/            ANTIGRAVITY
├── docs/                    ANTIGRAVITY (CONTRACT.md: YOU)
├── .github/workflows/       ANTIGRAVITY
└── data/                    git-ignored except data/outputs/sample/
    ├── raw/                 backend only
    └── outputs/<run_id>/    backend writes, report reads
```

## 2. Monitored Tickers
| Exchange | Ticker | Name | Social Sources |
| :--- | :--- | :--- | :--- |
| **HKEX** | `0700.HK` | Tencent | Futu/Moomoo + AAStocks (backup: LIHKG) |
| **HKEX** | `9988.HK` | Alibaba | Futu/Moomoo + AAStocks (backup: LIHKG) |
| **HKEX** | `3690.HK` | Meituan | Futu/Moomoo + AAStocks (backup: LIHKG) |
| **HKEX** | `1810.HK` | Xiaomi | Futu/Moomoo + AAStocks (backup: LIHKG) |
| **HKEX** | `1211.HK` | BYD | Futu/Moomoo + AAStocks (backup: LIHKG) |
| **KRX** | `005930.KS` | Samsung Electronics | Naver Finance + Daum Finance (backup: Blind) |
| **KRX** | `000660.KS` | SK Hynix | Naver Finance + Daum Finance (backup: Blind) |
| **KRX** | `035420.KS` | NAVER | Naver Finance + Daum Finance (backup: Blind) |
| **KRX** | `035720.KS` | Kakao | Naver Finance + Daum Finance (backup: Blind) |
| **KRX** | `005380.KS` | Hyundai Motor | Naver Finance + Daum Finance (backup: Blind) |

## 3. Output Files in `data/outputs/<run_id>/` (and `data/outputs/latest/`)
1. **`posts.csv`**: `post_id`, `exchange`, `ticker`, `source`, `created_at_utc`, `text`, `likes`, `reposts`, `views`, `author_followers`, `url`, `sentiment_score` (-1..1), `sentiment_label` (bullish/bearish/neutral), `weight`, `bar_date`.
2. **`news.csv`**: `ticker`, `title`, `url`, `published_at_utc`, `source_domain`, `summary`, `sentiment_score`, `bar_date`.
3. **`prices.csv`**: `ticker`, `date`, `open`, `high`, `low`, `close`, `volume`, `currency` (at least 90 trading days).
4. **`daily.csv`**: `ticker`, `date`, `mentions`, `social_sent_mean`, `social_sent_weighted`, `bullish_ratio`, `news_count`, `news_sent_mean`, `close`, `return_1d`.
5. **`summary.json`**: a list with one object per ticker:
```json
{
  "exchange": "HKEX", "ticker": "0700.HK", "name": "Tencent",
  "currency": "HKD", "current_price": 412.6, "as_of": "2026-11-20",
  "mentions_10d": 1843, "mentions_change_pct": 22.5,
  "sentiment_score": 0.31, "news_sentiment": 0.12,
  "direction": "bullish", "confidence": 0.64,
  "support": [398.0, 385.5], "resistance": [421.0, 436.8],
  "position": {"side": "long", "entry": 412.6, "stop": 397.0, "target": 435.0, "risk_reward": 1.44},
  "prediction": {"target_date": "2026-11-21", "predicted_close": 416.1, "low": 407.9, "high": 424.3},
  "top_posts": [{"source": "futu", "text": "...", "sentiment_score": 0.8, "url": "..."}],
  "top_news": [{"title": "...", "url": "...", "sentiment_score": 0.2}]
}
```
6. **`metrics.json`**: `{"overall": {...}, "per_ticker": {...}}`. Each entry has `mae`, `rmse`, `mape`, `directional_accuracy` and `n_predictions`, for the model, the naive baseline and the no-sentiment model.
7. **`predictions_log.csv`**: `run_id`, `ticker`, `target_date`, `predicted_close`, `actual_close`, `error`, `error_pct`, `direction_hit`.

## 4. Shared Ingestion & Modeling Rules
• **Alignment**: a post or article counts toward the next trading bar that hasn't closed yet. HKEX closes 08:00 UTC, KRX 06:30 UTC.
• **Direction**: score > 0.15 bullish, < -0.15 bearish, else neutral.
• **Missing values**: 0 for counts, null for unknown numbers. Never invent values.
• **Secrets**: only in .env, never committed.
• **Git**: Claude Code works on backend. Antigravity handles git commits.
