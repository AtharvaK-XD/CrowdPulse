# CrowdPulse — Multi-Exchange Retail Sentiment & Predictive Market Intelligence

CrowdPulse is an autonomous market intelligence and predictive analytics platform monitoring retail investor sentiment, financial news momentum, dynamic technical support/resistance levels, and predictive close pricing for **10 major equities** across the **Hong Kong Stock Exchange (HKEX)** and **Korea Exchange (KRX)**.

---

## 1. Monitored Universe

| Exchange | Ticker | Company Name | Currency | Close (UTC) | Primary Social Sources | News Domain Coverage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **HKEX** | `0700.HK` | Tencent Holdings | HKD | 08:00 UTC | Futu / Moomoo, LIHKG | SCMP, Reuters, Bloomberg, Caixin |
| **HKEX** | `9988.HK` | Alibaba Group | HKD | 08:00 UTC | Futu / Moomoo, LIHKG | SCMP, Reuters, Bloomberg, Caixin |
| **HKEX** | `3690.HK` | Meituan | HKD | 08:00 UTC | Futu / Moomoo, LIHKG | SCMP, Reuters, Bloomberg, Caixin |
| **HKEX** | `1810.HK` | Xiaomi Corp | HKD | 08:00 UTC | Futu / Moomoo, LIHKG | SCMP, Reuters, Bloomberg, Caixin |
| **HKEX** | `1211.HK` | BYD Company | HKD | 08:00 UTC | Futu / Moomoo, LIHKG | SCMP, Reuters, Bloomberg, Caixin |
| **KRX** | `005930.KS` | Samsung Electronics | KRW | 06:30 UTC | Naver Finance, Daum Finance | Yonhap, Korea Herald, KED Global, Maeil |
| **KRX** | `000660.KS` | SK Hynix | KRW | 06:30 UTC | Naver Finance, Daum Finance | Yonhap, Korea Herald, KED Global, Maeil |
| **KRX** | `035420.KS` | NAVER Corp | KRW | 06:30 UTC | Naver Finance, Daum Finance | Yonhap, Korea Herald, KED Global, Maeil |
| **KRX** | `035720.KS` | Kakao Corp | KRW | 06:30 UTC | Naver Finance, Daum Finance | Yonhap, Korea Herald, KED Global, Maeil |
| **KRX** | `005380.KS` | Hyundai Motor | KRW | 06:30 UTC | Naver Finance, Daum Finance | Yonhap, Korea Herald, KED Global, Maeil |

---

## 2. Key Architecture & Quantitative Principles

1. **Temporal Market Alignment & Look-Ahead Prevention**:
   - Posts and news published on or before market close (HKEX: `08:00 UTC`; KRX: `06:30 UTC`) are assigned to day $T$.
   - Any content published after market close, over the weekend, or across exchange holidays rolls forward into the next open trading bar ($T+1$).
2. **Hybrid Sentiment Scoring**:
   - Scores blend localized Chinese and Korean financial sentiment lexicons **50/50** with transformer classification models.
   - Requires $\ge 5$ posts per ticker to formulate a directional shift.
   - **Relative Baseline Comparison**: Direction evaluates recent sentiment relative to the ticker's historical baseline (z-score), rather than relying on a static cutoff.
3. **Dynamic Support & Resistance ($S_1, S_2, R_1, R_2$)**:
   - Computed from rolling 30-day swing extrema and volatility clusters, enforcing $S_2 < S_1 \le P_0 \le R_1 < R_2$.
4. **Suggested Trade Positioning & Risk Management**:
   - Long and short targets target a minimum Risk/Reward ratio of $1.30$.
   - No position (`none`) is recommended if the model forecast and sentiment direction disagree, or if conviction is below threshold.
   - Compliance notice automatically appended to short ideas: *"avoid / hedge: retail short selling is restricted in Hong Kong and Korea"*.
5. **Predictive Close Modeling**:
   - Multivariate pooled **Ridge Regression** with L2 regularization ($\alpha = 1.0$) predicting next-day close return ($n+1$), bounded at $\pm 3\times$ the daily ATR move.
6. **Self-Contained Offline HTML Report**:
   - Pure standalone HTML (`report.html`) containing inline Plotly JavaScript and CSS, requiring no external CDN connections at runtime.

---

## 3. Quickstart & Installation

### Requirements
- Python 3.11+
- Virtual environment recommended

```bash
git clone https://github.com/AtharvaK-XD/CrowdPulse.git
cd CrowdPulse
pip install -r requirements.txt
```

### Run Tests (100% Offline)
```bash
pytest tests/ -v
```
All 125 tests run completely offline with mocked networks.

### Build Visual Report
Generate the self-contained offline HTML report and high-resolution leaderboard image:
```bash
# Using offline sample fixtures:
python run.py --sample

# Or directly:
python report/build_report.py --run data/outputs/sample --output report.html --img leaderboard.png
```

### Full Live Pipeline Run
With `APIFY_TOKEN` and `EXA_API_KEY` configured in `.env`:
```bash
python run.py
```
Or run the pipeline CLI directly:
```bash
python scripts/run_pipeline.py
```

---

## 4. Repository Structure

```
CrowdPulse/
├── README.md                # Project documentation & execution guide
├── config.yaml              # Exchange settings, tickers, and data sources
├── requirements.txt         # Core dependencies
├── run.py                   # Unified pipeline -> report execution entrypoint
├── crowdpulse/              # Core backend package
│   ├── analytics/           # Support/resistance levels & positioning
│   ├── features/            # Temporal bar date alignment & daily aggregates
│   ├── ingest/              # Multi-source scrapers & API clients
│   ├── model/               # Ridge regression, walk-forward evaluation, prediction
│   └── nlp/                 # Lexicons & hybrid sentiment scoring
├── report/                  # Visual HTML report builder & Plotly charts
│   ├── build_report.py      # Standalone report compiler CLI
│   ├── charts.py            # Plotly interactive chart builders
│   └── templates/base.html  # Self-contained dark-theme Jinja2 template
├── fixtures/                # Synthetic fixture generator (fixed seed)
├── data/outputs/sample/     # Committed sample data for offline grading & CI
├── docs/
│   ├── CONTRACT.md          # Shared schema contract across agents
│   ├── ACTORS.md            # Apify actor research and pricing breakdown
│   └── METHODOLOGY.md       # Quantitative framework specification
└── tests/                   # 125 offline unit & contract tests
```

---

## 5. Continuous Integration (CI)
GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push and pull request:
- Validates the complete 125-test suite offline.
- Generates synthetic fixture outputs adhering to `docs/CONTRACT.md`.
- Compiles the self-contained HTML report and exports `leaderboard.png`.
- Uploads the visual report as a downloadable artifact.
