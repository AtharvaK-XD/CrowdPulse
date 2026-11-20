"""Fake data generator for CrowdPulse.

Generates realistic fake output files adhering strictly to docs/CONTRACT.md.
Creates sample data in data/outputs/sample/ for offline development, QA, and CI.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
from datetime import datetime, timedelta

FIXED_SEED = 42

TICKERS_INFO = [
    # HKEX
    {
        "exchange": "HKEX",
        "ticker": "0700.HK",
        "name": "Tencent",
        "currency": "HKD",
        "base_price": 412.6,
        "direction": "bullish",
        "sources": ["futu", "aastocks", "lihkg"],
        "edge_case": "long_post",
    },
    {
        "exchange": "HKEX",
        "ticker": "9988.HK",
        "name": "Alibaba",
        "currency": "HKD",
        "base_price": 85.4,
        "direction": "neutral",
        "sources": ["futu", "aastocks"],
        "edge_case": None,
    },
    {
        "exchange": "HKEX",
        "ticker": "3690.HK",
        "name": "Meituan",
        "currency": "HKD",
        "base_price": 168.2,
        "direction": "bearish",
        "sources": ["futu", "aastocks"],
        "edge_case": None,
    },
    {
        "exchange": "HKEX",
        "ticker": "1810.HK",
        "name": "Xiaomi",
        "currency": "HKD",
        "base_price": 28.5,
        "direction": "bullish",
        "sources": ["futu"],  # EDGE CASE: Single source
        "edge_case": "single_source",
    },
    {
        "exchange": "HKEX",
        "ticker": "1211.HK",
        "name": "BYD",
        "currency": "HKD",
        "base_price": 265.0,
        "direction": "bullish",
        "sources": ["futu", "aastocks"],
        "edge_case": None,
    },
    # KRX
    {
        "exchange": "KRX",
        "ticker": "005930.KS",
        "name": "Samsung Electronics",
        "currency": "KRW",
        "base_price": 56800.0,
        "direction": "bearish",
        "sources": ["naver", "daum"],
        "edge_case": None,
    },
    {
        "exchange": "KRX",
        "ticker": "000660.KS",
        "name": "SK Hynix",
        "currency": "KRW",
        "base_price": 172500.0,
        "direction": "bullish",
        "sources": ["naver", "daum"],
        "edge_case": None,
    },
    {
        "exchange": "KRX",
        "ticker": "035420.KS",
        "name": "NAVER",
        "currency": "KRW",
        "base_price": 175500.0,
        "direction": "neutral",
        "sources": ["naver", "daum"],
        "edge_case": "no_position",  # EDGE CASE: No position suggested
    },
    {
        "exchange": "KRX",
        "ticker": "035720.KS",
        "name": "Kakao",
        "currency": "KRW",
        "base_price": 38200.0,
        "direction": "bearish",
        "sources": ["naver", "daum", "blind"],
        "edge_case": "zero_news",  # EDGE CASE: Zero news in 10 days
    },
    {
        "exchange": "KRX",
        "ticker": "005380.KS",
        "name": "Hyundai Motor",
        "currency": "KRW",
        "base_price": 218000.0,
        "direction": "bullish",
        "sources": ["naver", "daum"],
        "edge_case": "null_prediction",  # EDGE CASE: Model failed / null prediction
    },
]

CHINESE_SAMPLE_TEXTS = {
    "bullish": [
        "腾讯Q3业绩超预期，微信视频号广告收入增长强劲，AI大模型逐步落地，现价极具吸引力，继续加仓！",
        "南向资金持续净流入，港股互联网板块估值处于历史低位，技术形态突破阻力位，看好后市表现。",
        "今日大单买入明显，主力资金持续吸筹，支撑位稳固，目标价看向更高区间。",
        "业绩发布会指引乐观，海外游戏扩张势头良好，现金流充沛，回购力度加大，强烈推荐持有。",
    ],
    "bearish": [
        "大盘走弱，外资流出压力较大，短期缺乏催化剂，上方套牢盘较重，建议减仓规避风险。",
        "宏观消费数据疲软，核心电商竞争激烈，利润率承压，跌破关键均线支撑，防范下行风险。",
        "技术形态形成死叉，成交量萎缩，短期难以反弹，逢高出局为主。",
        "政策监管预期仍有不确定性，估值修复面临天花板，后市保持谨慎观望态度。",
    ],
    "neutral": [
        "目前在箱体区间震荡整理，多空双方胶着，等待新的催化剂或财报指引，建议观望。",
        "今日成交清淡，跟随大盘窄幅波动，短期方向尚不明朗，维持中性评级。",
        "估值合理，股息率有一定防御性，但在缺乏增量资金前难有大行情，持股观望。",
    ],
}

KOREAN_SAMPLE_TEXTS = {
    "bullish": [
        "외국인과 기관의 강력한 순매수 유입 중. 실적 턴어라운드 본격화되며 HBM 및 메모리 반도체 수요 폭발적 증가 예상.",
        "차트상 20일선 골든크로스 발생! 강력한 지지선 확인되었고 다음 목표가까지 홀딩합니다.",
        "AI 반도체 밸류체인 독점적 지위 유지 중. 내년 영업이익 컨센서스 상향 조정 소식에 매수세 집중.",
        "바닥 확인 완료. 분할 매수로 대응하면 연말까지 유의미한 수익률 기대해볼 만함.",
    ],
    "bearish": [
        "글로벌 경기 둔화 우려와 반도체 업황 피크아웃 논란 지속. 외인 매도세 강해 단기 지지선 이탈 위험 큼.",
        "실적 가이던스 하향 조정 충격. 개인들만 물량 받는 중이라 추가 하락 가능성 열어둬야 합니다.",
        "기술적으로 헤드앤숄더 패턴 완성 단계. 손절 라인 엄수하고 리스크 관리 필수.",
        "공매도 잔고 증가 및 환율 불안으로 외국인 수급 이탈 심화. 당분간 관망 추천.",
    ],
    "neutral": [
        "현재 박스권 하단에서 횡보 중. 뚜렷한 모멘텀이 없어 당분간 좁은 레인지 매매가 유효할 듯.",
        "실적 발표 앞두고 눈치보기 장세 이어짐. 호재와 악재가 팽팽히 맞서는 중.",
        "거래량 부족으로 방향성 탐색 국면. 무리한 진입보다는 확인 매매가 바람직해 보임.",
    ],
}

SAMPLE_NEWS = {
    "HKEX": [
        ("Tech giant reports double-digit cloud revenue growth in Q3", "scmp.com"),
        ("Hang Seng Tech Index rallies as overseas capital flows return", "bloomberg.com"),
        ("New regulatory guidelines clarify cross-border data transfer framework", "reuters.com"),
        ("Autonomous driving rollout accelerates across tier-1 cities", "caixinglobal.com"),
        ("Brokerage upgrades sector outlook citing robust buyback programs", "aastocks.com"),
    ],
    "KRX": [
        ("Semiconductor export volume hits fresh high on AI server boom", "kedglobal.com"),
        ("Bank of Korea holds benchmark rate steady amid currency fluctuations", "yonhapnews.co.kr"),
        ("Automaker expands global EV production hub with solid-state investment", "koreaherald.com"),
        ("Domestic AI platform secures massive cloud enterprise contracts", "etnews.com"),
        ("Foreign investors snap up large-cap tech shares on valuation appeal", "mk.co.kr"),
    ],
}

LONG_POST_SAMPLE = (
    "【深度研报与筹码结构复盘】"
    "结合近10个交易日的盘面特征与社交情绪演变，当前标的正处于筹码换手与技术形态突破的关键临界点。"
    "首先，从基本面观察，核心业务基本盘持续稳固，经营性现金流保持健康充沛，近两季度研发费用转化为高附加值AI服务的路径已经清晰验证。"
    "其次，在资金结构方面，近期南向资金与机构席位连续呈现净流入格局，散户恐慌盘在前期震荡中已被有效清洗，持股集中度显著提升。"
    "从衍生品与隐含波动率结构来看，看涨期权持仓量大幅提升，市场对于后市预期已由防御转为进攻。"
    "技术面上，30日均线与60日均线形成多头排列，量能温和放大，下方第一支撑位有效经受多次回踩考验，形成坚实的双重底颈线结构。"
    "综合风险收益比，建议以支撑位为下行止损参照，分批建仓做多，目标位看至上方密集阻力区。"
    "需密切跟踪海外流动性外溢效应及美元指数变动对亚太权益市场的扰动。"
)


def generate_fixtures(output_dir: Path, as_of_date: str = "2026-11-20") -> None:
    random.seed(FIXED_SEED)
    output_dir.mkdir(parents=True, exist_ok=True)

    as_of = datetime.strptime(as_of_date, "%Y-%m-%d").date()
    target_date = (as_of + timedelta(days=1)).strftime("%Y-%m-%d")

    # 1. Generate 90+ trading days of prices (100 trading days)
    trading_dates: list[str] = []
    curr = as_of
    while len(trading_dates) < 100:
        if curr.weekday() < 5:  # Monday to Friday
            trading_dates.append(curr.strftime("%Y-%m-%d"))
        curr -= timedelta(days=1)
    trading_dates.reverse()

    prices_rows = []
    ticker_price_series = {}

    for tinfo in TICKERS_INFO:
        ticker = tinfo["ticker"]
        currency = tinfo["currency"]
        base = tinfo["base_price"]

        prices = []
        p = base * 0.90  # start slightly lower 100 days ago
        for dt in trading_dates:
            daily_pct = random.gauss(0.001, 0.015)
            open_p = p * (1 + random.gauss(0, 0.005))
            high_p = max(open_p, open_p * (1 + abs(random.gauss(0, 0.01))))
            low_p = min(open_p, open_p * (1 - abs(random.gauss(0, 0.01))))
            close_p = open_p * (1 + daily_pct)
            high_p = max(high_p, close_p)
            low_p = min(low_p, close_p)
            vol = int(random.lognormvariate(14, 0.8)) if currency == "HKD" else int(random.lognormvariate(13, 0.8))

            round_dec = 2 if currency == "HKD" else 0
            open_val = round(open_p, round_dec)
            high_val = round(high_p, round_dec)
            low_val = round(low_p, round_dec)
            close_val = round(close_p, round_dec)
            if round_dec == 0:
                open_val, high_val, low_val, close_val = int(open_val), int(high_val), int(low_val), int(close_val)

            prices_rows.append({
                "ticker": ticker,
                "date": dt,
                "open": open_val,
                "high": high_val,
                "low": low_val,
                "close": close_val,
                "volume": vol,
                "currency": currency,
            })
            prices.append(close_val)
            p = close_p
        ticker_price_series[ticker] = prices

    # Write prices.csv
    prices_path = output_dir / "prices.csv"
    with open(prices_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ticker", "date", "open", "high", "low", "close", "volume", "currency"])
        writer.writeheader()
        writer.writerows(prices_rows)

    # 2. Generate posts.csv and news.csv over 10 days
    last_10_dates = trading_dates[-10:]
    posts_rows = []
    news_rows = []
    daily_stats = {t["ticker"]: {dt: {"mentions": 0, "scores": [], "weights": [], "bullish_count": 0, "news_scores": []} for dt in last_10_dates} for t in TICKERS_INFO}

    post_counter = 1000

    for tinfo in TICKERS_INFO:
        ticker = tinfo["ticker"]
        exchange = tinfo["exchange"]
        direction = tinfo["direction"]
        sources = tinfo["sources"]
        edge = tinfo["edge_case"]

        # Number of posts for this ticker
        n_posts = random.randint(70, 250)

        for i in range(n_posts):
            post_counter += 1
            source = random.choice(sources)
            bdate = random.choice(last_10_dates)

            # Generate UTC timestamp for that bar date
            bar_d = datetime.strptime(bdate, "%Y-%m-%d").date()
            hour = random.randint(0, 7) if exchange == "HKEX" else random.randint(0, 5)
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            created_at_utc = f"{bdate}T{hour:02d}:{minute:02d}:{second:02d}Z"

            # Sentiment score based on overall ticker direction
            if direction == "bullish":
                score = round(random.triangular(-0.1, 0.9, 0.45), 3)
            elif direction == "bearish":
                score = round(random.triangular(-0.9, 0.1, -0.45), 3)
            else:
                score = round(random.triangular(-0.35, 0.35, 0.0), 3)

            if score > 0.15:
                label = "bullish"
            elif score < -0.15:
                label = "bearish"
            else:
                label = "neutral"

            likes = int(random.expovariate(1 / 25))
            reposts = int(likes * random.uniform(0.05, 0.3))
            views = int(likes * random.uniform(10, 80)) + random.randint(50, 500)
            author_followers = int(random.lognormvariate(7, 1.2))

            # Weight calculation heuristic: engagement + follower influence
            weight = round(1.0 + math.log1p(likes * 2 + views * 0.1 + author_followers * 0.01), 2)

            # Text selection
            if exchange == "HKEX":
                text_pool = CHINESE_SAMPLE_TEXTS[label]
            else:
                text_pool = KOREAN_SAMPLE_TEXTS[label]
            post_text = random.choice(text_pool)

            # Edge case: long post
            if edge == "long_post" and i == 0:
                post_text = LONG_POST_SAMPLE

            post_id = f"{source}_{post_counter}"
            url = f"https://{source}.example.com/post/{post_counter}"

            posts_rows.append({
                "post_id": post_id,
                "exchange": exchange,
                "ticker": ticker,
                "source": source,
                "created_at_utc": created_at_utc,
                "text": post_text,
                "likes": likes,
                "reposts": reposts,
                "views": views,
                "author_followers": author_followers,
                "url": url,
                "sentiment_score": score,
                "sentiment_label": label,
                "weight": weight,
                "bar_date": bdate,
            })

            daily_stats[ticker][bdate]["mentions"] += 1
            daily_stats[ticker][bdate]["scores"].append(score)
            daily_stats[ticker][bdate]["weights"].append(weight)
            if label == "bullish":
                daily_stats[ticker][bdate]["bullish_count"] += 1

        # Generate News (unless edge case zero_news)
        if edge != "zero_news":
            n_news = random.randint(4, 15)
            for j in range(n_news):
                bdate = random.choice(last_10_dates)
                title_template, domain = random.choice(SAMPLE_NEWS[exchange])
                title = f"{tinfo['name']}: {title_template}"
                summary = f"Recent market coverage regarding {tinfo['name']} ({ticker}) operations, financial posture, and market performance."
                hour = random.randint(0, 6)
                pub_utc = f"{bdate}T{hour:02d}:{random.randint(0, 59):02d}:00Z"

                if direction == "bullish":
                    n_score = round(random.triangular(0.0, 0.8, 0.35), 3)
                elif direction == "bearish":
                    n_score = round(random.triangular(-0.8, 0.0, -0.35), 3)
                else:
                    n_score = round(random.triangular(-0.25, 0.25, 0.0), 3)

                news_url = f"https://{domain}/article/{ticker.replace('.', '_')}_{j+100}"
                news_rows.append({
                    "ticker": ticker,
                    "title": title,
                    "url": news_url,
                    "published_at_utc": pub_utc,
                    "source_domain": domain,
                    "summary": summary,
                    "sentiment_score": n_score,
                    "bar_date": bdate,
                })
                daily_stats[ticker][bdate]["news_scores"].append(n_score)

    # Write posts.csv
    posts_path = output_dir / "posts.csv"
    with open(posts_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "post_id", "exchange", "ticker", "source", "created_at_utc", "text",
            "likes", "reposts", "views", "author_followers", "url",
            "sentiment_score", "sentiment_label", "weight", "bar_date"
        ])
        writer.writeheader()
        writer.writerows(posts_rows)

    # Write news.csv
    news_path = output_dir / "news.csv"
    with open(news_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "ticker", "title", "url", "published_at_utc", "source_domain", "summary",
            "sentiment_score", "bar_date"
        ])
        writer.writeheader()
        writer.writerows(news_rows)

    # 3. Generate daily.csv
    daily_rows = []
    # Build map of (ticker, date) -> close price from prices_rows
    close_price_map = { (r["ticker"], r["date"]): r["close"] for r in prices_rows }

    for tinfo in TICKERS_INFO:
        ticker = tinfo["ticker"]
        for idx, dt in enumerate(last_10_dates):
            st = daily_stats[ticker][dt]
            mentions = st["mentions"]
            if mentions > 0:
                s_mean = round(sum(st["scores"]) / mentions, 3)
                w_sum = sum(st["weights"])
                s_weighted = round(sum(s * w for s, w in zip(st["scores"], st["weights"])) / w_sum, 3)
                b_ratio = round(st["bullish_count"] / mentions, 3)
            else:
                s_mean = 0.0
                s_weighted = 0.0
                b_ratio = 0.0

            n_count = len(st["news_scores"])
            n_mean = round(sum(st["news_scores"]) / n_count, 3) if n_count > 0 else 0.0

            curr_close = close_price_map.get((ticker, dt), tinfo["base_price"])
            # Previous day close
            prev_dt = trading_dates[trading_dates.index(dt) - 1]
            prev_close = close_price_map.get((ticker, prev_dt), curr_close)
            ret_1d = round((curr_close - prev_close) / prev_close, 4)

            daily_rows.append({
                "ticker": ticker,
                "date": dt,
                "mentions": mentions,
                "social_sent_mean": s_mean,
                "social_sent_weighted": s_weighted,
                "bullish_ratio": b_ratio,
                "news_count": n_count,
                "news_sent_mean": n_mean,
                "close": curr_close,
                "return_1d": ret_1d,
            })

    # Write daily.csv
    daily_path = output_dir / "daily.csv"
    with open(daily_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "ticker", "date", "mentions", "social_sent_mean", "social_sent_weighted",
            "bullish_ratio", "news_count", "news_sent_mean", "close", "return_1d"
        ])
        writer.writeheader()
        writer.writerows(daily_rows)

    # 4. Generate summary.json
    summary_list = []
    for tinfo in TICKERS_INFO:
        ticker = tinfo["ticker"]
        exchange = tinfo["exchange"]
        name = tinfo["name"]
        currency = tinfo["currency"]
        direction = tinfo["direction"]
        edge = tinfo["edge_case"]

        # Aggregate 10-day stats
        ticker_posts = [p for p in posts_rows if p["ticker"] == ticker]
        mentions_10d = len(ticker_posts)
        mentions_change_pct = round(random.uniform(-15.0, 35.0), 1)

        if mentions_10d > 0:
            sent_score = round(sum(p["sentiment_score"] for p in ticker_posts) / mentions_10d, 2)
        else:
            sent_score = 0.0

        ticker_news = [n for n in news_rows if n["ticker"] == ticker]
        if ticker_news:
            news_sent = round(sum(n["sentiment_score"] for n in ticker_news) / len(ticker_news), 2)
        else:
            news_sent = 0.0

        curr_price = close_price_map.get((ticker, as_of_date), tinfo["base_price"])
        dec = 2 if currency == "HKD" else 0

        # Support & Resistance levels
        s1 = round(curr_price * 0.965, dec)
        s2 = round(curr_price * 0.935, dec)
        r1 = round(curr_price * 1.035, dec)
        r2 = round(curr_price * 1.070, dec)
        if dec == 0:
            s1, s2, r1, r2 = int(s1), int(s2), int(r1), int(r2)

        # Position
        if edge == "no_position":
            position = {
                "side": "none",
                "entry": None,
                "stop": None,
                "target": None,
                "risk_reward": None,
            }
            confidence = 0.42
        elif direction == "bullish":
            stop_val = s1
            target_val = r1
            risk = max(curr_price - stop_val, 0.01)
            reward = max(target_val - curr_price, 0.01)
            rr = round(reward / risk, 2)
            position = {
                "side": "long",
                "entry": curr_price,
                "stop": stop_val,
                "target": target_val,
                "risk_reward": rr,
            }
            confidence = round(random.uniform(0.60, 0.78), 2)
        elif direction == "bearish":
            stop_val = r1
            target_val = s1
            risk = max(stop_val - curr_price, 0.01)
            reward = max(curr_price - target_val, 0.01)
            rr = round(reward / risk, 2)
            position = {
                "side": "short",
                "entry": curr_price,
                "stop": stop_val,
                "target": target_val,
                "risk_reward": rr,
            }
            confidence = round(random.uniform(0.58, 0.75), 2)
        else:  # neutral
            position = {
                "side": "none",
                "entry": None,
                "stop": None,
                "target": None,
                "risk_reward": None,
            }
            confidence = round(random.uniform(0.40, 0.52), 2)

        # Prediction
        if edge == "null_prediction":
            prediction = {
                "target_date": target_date,
                "predicted_close": None,
                "low": None,
                "high": None,
            }
        else:
            delta_pct = random.uniform(0.005, 0.02) if direction == "bullish" else (random.uniform(-0.02, -0.005) if direction == "bearish" else random.uniform(-0.004, 0.004))
            pred_close = round(curr_price * (1 + delta_pct), dec)
            p_low = round(pred_close * 0.985, dec)
            p_high = round(pred_close * 1.015, dec)
            if dec == 0:
                pred_close, p_low, p_high = int(pred_close), int(p_low), int(p_high)
            prediction = {
                "target_date": target_date,
                "predicted_close": pred_close,
                "low": p_low,
                "high": p_high,
            }

        # Top posts and top news
        sorted_posts = sorted(ticker_posts, key=lambda x: (x["likes"] + x["views"] * 0.1), reverse=True)
        top_posts = [
            {
                "source": p["source"],
                "text": p["text"][:140] + ("..." if len(p["text"]) > 140 else ""),
                "sentiment_score": p["sentiment_score"],
                "url": p["url"],
            }
            for p in sorted_posts[:3]
        ]

        top_news = [
            {
                "title": n["title"],
                "url": n["url"],
                "sentiment_score": n["sentiment_score"],
            }
            for n in ticker_news[:3]
        ]

        summary_list.append({
            "exchange": exchange,
            "ticker": ticker,
            "name": name,
            "currency": currency,
            "current_price": curr_price,
            "as_of": as_of_date,
            "mentions_10d": mentions_10d,
            "mentions_change_pct": mentions_change_pct,
            "sentiment_score": sent_score,
            "news_sentiment": news_sent,
            "direction": direction,
            "confidence": confidence,
            "support": [s1, s2],
            "resistance": [r1, r2],
            "position": position,
            "prediction": prediction,
            "top_posts": top_posts,
            "top_news": top_news,
        })

    # Write summary.json
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_list, f, indent=2, ensure_ascii=False)

    # 5. Generate metrics.json
    def make_metric_entry(n: int, factor: float = 1.0) -> dict:
        return {
            "mae": round(factor * 3.42, 2),
            "rmse": round(factor * 4.88, 2),
            "mape": round(factor * 0.0165, 4),
            "directional_accuracy": round(max(0.40, min(0.85, 0.65 / factor)), 3),
            "n_predictions": n,
        }

    metrics_obj = {
        "overall": {
            "model": make_metric_entry(100, factor=1.0),
            "naive": make_metric_entry(100, factor=1.38),
            "no_sentiment": make_metric_entry(100, factor=1.19),
        },
        "per_ticker": {
            t["ticker"]: {
                "model": make_metric_entry(10, factor=1.0),
                "naive": make_metric_entry(10, factor=1.35),
                "no_sentiment": make_metric_entry(10, factor=1.18),
            }
            for t in TICKERS_INFO
        },
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_obj, f, indent=2)

    # 6. Generate predictions_log.csv
    pred_log_rows = []
    for dt in last_10_dates:
        for tinfo in TICKERS_INFO:
            ticker = tinfo["ticker"]
            currency = tinfo["currency"]
            dec = 2 if currency == "HKD" else 0
            act_close = close_price_map.get((ticker, dt), tinfo["base_price"])
            err_pct = round(random.gauss(0.001, 0.012), 4)
            pred_val = round(act_close * (1 + err_pct), dec)
            if dec == 0:
                pred_val = int(pred_val)
            err = round(pred_val - act_close, dec)
            if dec == 0:
                err = int(err)
            hit = 1 if (err_pct >= 0 and tinfo["direction"] == "bullish") or (err_pct < 0 and tinfo["direction"] == "bearish") or (abs(err_pct) < 0.01) else 0

            pred_log_rows.append({
                "run_id": "run_20261120_sample",
                "ticker": ticker,
                "target_date": dt,
                "predicted_close": pred_val,
                "actual_close": act_close,
                "error": err,
                "error_pct": err_pct,
                "direction_hit": hit,
            })

    pred_log_path = output_dir / "predictions_log.csv"
    with open(pred_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run_id", "ticker", "target_date", "predicted_close", "actual_close",
            "error", "error_pct", "direction_hit"
        ])
        writer.writeheader()
        writer.writerows(pred_log_rows)

    print(f"Successfully generated all contract files in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample fixture data for CrowdPulse")
    parser.add_argument("--output-dir", type=str, default="data/outputs/sample", help="Target output directory")
    parser.add_argument("--as-of", type=str, default="2026-11-20", help="As of date (YYYY-MM-DD)")
    args = parser.parse_args()

    out_p = Path(args.output_dir)
    generate_fixtures(out_p, args.as_of)


if __name__ == "__main__":
    main()
