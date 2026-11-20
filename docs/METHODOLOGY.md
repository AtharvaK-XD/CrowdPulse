# CrowdPulse — Methodology & Quantitative Framework

**Version:** 1.0.0  
**Effective Date:** 2026-11-20  
**Authors:** CrowdPulse Team (Claude Code & Antigravity)  

---

## 1. Executive Summary

CrowdPulse is an autonomous market intelligence system that monitors retail sentiment, news flow, and technical price structure across 10 major Asian equities listed on the Hong Kong Stock Exchange (**HKEX**) and Korea Exchange (**KRX**). 

The platform aggregates social discussion from primary regional retail platforms, extracts news from global and local financial outlets, applies a weighted sentiment scoring model, calculates dynamic support/resistance zones, and produces next-day (n+1) close predictions with strict walk-forward evaluation.

---

## 2. Monitored Universe

| Exchange | Ticker | Company Name | Currency | Trading Hours (Local) | Daily Close (UTC) | Primary Social Sources | News Domain Coverage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **HKEX** | `0700.HK` | Tencent Holdings | HKD | 09:30–16:00 HKT | **08:00 UTC** | Futu/Moomoo, AAStocks, LIHKG | SCMP, Reuters, Bloomberg, Caixin |
| **HKEX** | `9988.HK` | Alibaba Group | HKD | 09:30–16:00 HKT | **08:00 UTC** | Futu/Moomoo, AAStocks, LIHKG | SCMP, Reuters, Bloomberg, Caixin |
| **HKEX** | `3690.HK` | Meituan | HKD | 09:30–16:00 HKT | **08:00 UTC** | Futu/Moomoo, AAStocks, LIHKG | SCMP, Reuters, Bloomberg, Caixin |
| **HKEX** | `1810.HK` | Xiaomi Corp | HKD | 09:30–16:00 HKT | **08:00 UTC** | Futu/Moomoo, AAStocks, LIHKG | SCMP, Reuters, Bloomberg, Caixin |
| **HKEX** | `1211.HK` | BYD Company | HKD | 09:30–16:00 HKT | **08:00 UTC** | Futu/Moomoo, AAStocks, LIHKG | SCMP, Reuters, Bloomberg, Caixin |
| **KRX** | `005930.KS` | Samsung Electronics | KRW | 09:00–15:30 KST | **06:30 UTC** | Naver Finance, Daum Finance, Blind | Yonhap, Korea Herald, KED Global, Maeil |
| **KRX** | `000660.KS` | SK Hynix | KRW | 09:00–15:30 KST | **06:30 UTC** | Naver Finance, Daum Finance, Blind | Yonhap, Korea Herald, KED Global, Maeil |
| **KRX** | `035420.KS` | NAVER Corp | KRW | 09:00–15:30 KST | **06:30 UTC** | Naver Finance, Daum Finance, Blind | Yonhap, Korea Herald, KED Global, Maeil |
| **KRX** | `035720.KS` | Kakao Corp | KRW | 09:00–15:30 KST | **06:30 UTC** | Naver Finance, Daum Finance, Blind | Yonhap, Korea Herald, KED Global, Maeil |
| **KRX** | `005380.KS` | Hyundai Motor | KRW | 09:00–15:30 KST | **06:30 UTC** | Naver Finance, Daum Finance, Blind | Yonhap, Korea Herald, KED Global, Maeil |

---

## 3. Market Alignment & Look-Ahead Prevention

A fundamental flaw in naive sentiment pipelines is **look-ahead bias** (attributing posts or news published after market close to that day's session). CrowdPulse enforces strict temporal alignment:

1. **Market Cutoff Times:**
   - **HKEX:** Regular market trading closes at 16:00 HKT (`08:00:00 UTC`).
   - **KRX:** Regular market trading closes at 15:30 KST (`06:30:00 UTC`).
2. **Bar Date Assignment (`bar_date`):**
   - Any post or news published **at or before** the market close UTC timestamp on trading day $T$ is assigned to bar date $T$.
   - Any post or news published **after** the market close UTC timestamp on trading day $T$ is assigned to trading day $T+1$ (the next open trading bar).
   - Weekend posts (Saturday and Sunday) and exchange holidays are accumulated into Monday's trading bar.
3. **No-Look-Ahead Guarantee:**
   - Features generated for prediction of day $T+1$'s close use data strictly up to the close of day $T$.

---

## 4. Sentiment Scoring & Weighting Engine

### 4.1. Raw Score Normalization
- Sentiment scores are normalized bounded floats in the interval $[-1.0, +1.0]$:
  - $+1.0$: Strongest positive/bullish conviction.
  - $0.0$: Neutral, descriptive, or balanced sentiment.
  - $-1.0$: Strongest negative/bearish conviction.
- **Directional Categorization Rule:**
  $$\text{Direction} = \begin{cases} \text{Bullish}, & \text{score} > +0.15 \\ \text{Bearish}, & \text{score} < -0.15 \\ \text{Neutral}, & -0.15 \le \text{score} \le +0.15 \end{cases}$$

### 4.2. Engagement & Influence Weighting
Raw post volume can be heavily skewed by bots and low-quality spam. Each post receives a dynamic engagement weight $W_i$:
$$W_i = 1.0 + \ln\left(1 + 2.0 \cdot \text{likes} + 0.1 \cdot \text{views} + 0.01 \cdot \text{followers}\right)$$

Daily weighted sentiment for ticker $k$ on bar date $t$ is calculated as:
$$\bar{S}_{\text{weighted}}(k, t) = \frac{\sum_{i \in \text{posts}(k,t)} W_i \cdot s_i}{\sum_{i \in \text{posts}(k,t)} W_i}$$

---

## 5. Support & Resistance Formulations

Support and resistance levels represent supply and demand inflection points identified through rolling 30-day price extrema and volume-weighted clustering:

1. **First Support ($S_1$) & Resistance ($R_1$):**
   - $S_1$: Primary dynamic support level computed from the 20-day lowest swing close or lower Bollinger band ($-1.5\sigma$).
   - $R_1$: Primary dynamic resistance level computed from the 20-day highest swing close or upper Bollinger band ($+1.5\sigma$).
2. **Secondary Levels ($S_2, R_2$):**
   - $S_2$: Major support (30-day structural low).
   - $R_2$: Major resistance (30-day structural high).
3. **Validity Constraints:**
   - $S_2 < S_1 \le P_{\text{current}} \le R_1 < R_2$

---

## 6. Suggested Trade Positioning & Risk Management

CrowdPulse translates sentiment and technical levels into actionable risk-managed position parameters:

- **Long Position (Bullish Sentiment, Direction = `bullish`):**
  - **Entry:** Current Close $P_0$.
  - **Stop Loss:** Set at primary support $S_1$ (or $S_1 - 0.5\%$).
  - **Target:** Set at primary resistance $R_1$.
  - **Risk/Reward Ratio:** $\text{RR} = \frac{R_1 - P_0}{P_0 - S_1} \ge 1.30$.
- **Short Position (Bearish Sentiment, Direction = `bearish`):**
  - **Entry:** Current Close $P_0$.
  - **Stop Loss:** Set at primary resistance $R_1$.
  - **Target:** Set at primary support $S_1$.
  - **Risk/Reward Ratio:** $\text{RR} = \frac{P_0 - S_1}{R_1 - P_0} \ge 1.30$.
- **No Position (`side`: "none"):**
  - Triggered if Direction is `neutral`, confidence $< 0.55$, or if the risk/reward ratio is less than $1.20$.

---

## 7. Predictive Model & Walk-Forward Validation

### 7.1. Model Architecture
The next-day close prediction ($\hat{P}_{t+1}$) is estimated via a multivariate gradient-boosted regression pipeline combining:
1. **Price Momentum & Technical Features:** 5-day and 10-day returns, normalized ATR, MACD histogram, distance from 20-day moving average.
2. **Social Momentum Features:** 10-day mentions growth ($\Delta \text{vol}\%_{10d}$), volume z-score, daily weighted sentiment, bullish post ratio.
3. **News Sentiment Features:** 3-day news count and mean sentiment score.

### 7.2. Uncertainty Intervals
The predicted range $[\text{Low}_{t+1}, \text{High}_{t+1}]$ is derived from quantile regression at the 10th and 90th percentiles ($80\%$ empirical prediction interval).

### 7.3. Benchmark Comparisons (Scorecard)
To prove empirical validity, CrowdPulse benchmarks the model against two baselines across a 100-step walk-forward test window:
1. **Naive Last-Close Baseline:** $\hat{P}_{t+1} = P_t$.
2. **No-Sentiment Ablation Model:** Trained with price, volume, and technical indicators only (excluding all social and news features).

Evaluation metrics:
- **MAE:** $\frac{1}{N} \sum |P_{t+1} - \hat{P}_{t+1}|$
- **RMSE:** $\sqrt{\frac{1}{N} \sum (P_{t+1} - \hat{P}_{t+1})^2}$
- **MAPE:** $\frac{1}{N} \sum \left|\frac{P_{t+1} - \hat{P}_{t+1}}{P_{t+1}}\right|$
- **Directional Accuracy:** $\frac{1}{N} \sum \mathbf{1}_{\operatorname{sgn}(\hat{P}_{t+1} - P_t) = \operatorname{sgn}(P_{t+1} - P_t)}$

---

## 8. Limitations & Operational Constraints

1. **Retail Sentiment Reversal (Contrarian Risk):** Extreme euphoria (>0.85 sentiment with massive volume spikes) frequently coincides with retail blow-off tops. 
2. **Language Nuance & Slang:** Korean forum slang (e.g. 존버, 떡상, 떡락) and Cantonese internet colloquialisms require localized lexicons to avoid misclassification.
3. **Liquidity Asymmetry:** While large-cap equities (Tencent, Samsung) exhibit deep institutional liquidity dampening retail impact, mid-to-large growth names (Meituan, Kakao) demonstrate higher sensitivity to retail sentiment cascades.
4. **Offline Mode:** When running against sample fixtures or in CI environments, live API scrapers are bypassed and synthetic market dynamics adhering strictly to `docs/CONTRACT.md` are evaluated.
