"""Report builder CLI for CrowdPulse.

Compiles a single, self-contained, offline-compatible HTML report (report.html)
and exports an image snapshot (leaderboard.png) from pipeline output directories.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import sys
from typing import Any

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jinja2
import plotly.offline as po

from report.charts import (
    build_bubble_chart,
    build_candlestick_chart,
    build_volume_sentiment_chart,
    build_sources_chart,
    build_predictions_chart,
    export_leaderboard_image,
)


def load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def format_price(val: Any, currency: str = "") -> str:
    if val is None or val == "":
        return "—"
    try:
        f_val = float(val)
        if currency == "HKD":
            return f"{f_val:,.2f}"
        elif currency == "KRW":
            return f"{int(round(f_val)):,}"
        return f"{f_val:,.2f}"
    except (ValueError, TypeError):
        return str(val)


def format_int(val: Any) -> str:
    if val is None or val == "":
        return "0"
    try:
        return f"{int(val):,}"
    except (ValueError, TypeError):
        return str(val)


def build_report(run_dir: Path, output_html: Path, output_img: Path) -> None:
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    # 1. Load contract files
    posts = load_csv(run_dir / "posts.csv")
    news = load_csv(run_dir / "news.csv")
    prices = load_csv(run_dir / "prices.csv")
    daily = load_csv(run_dir / "daily.csv")
    summary = load_json(run_dir / "summary.json") or []
    metrics = load_json(run_dir / "metrics.json") or {"overall": {}, "per_ticker": {}}
    predictions_log = load_csv(run_dir / "predictions_log.csv")

    # 2. Split summary by exchange
    hkex_summaries = [s for s in summary if s.get("exchange") == "HKEX"]
    krx_summaries = [s for s in summary if s.get("exchange") == "KRX"]

    # Sort default by mentions
    hkex_summaries.sort(key=lambda x: int(x.get("mentions_10d", 0)), reverse=True)
    krx_summaries.sort(key=lambda x: int(x.get("mentions_10d", 0)), reverse=True)

    # 3. Generate individual charts for ticker cards
    for s in summary:
        t = s["ticker"]
        s["candlestick_chart_html"] = build_candlestick_chart(t, prices, s)
        s["volume_sentiment_chart_html"] = build_volume_sentiment_chart(t, daily)

    # 4. Generate section aggregate charts
    hkex_tickers = [s["ticker"] for s in hkex_summaries]
    krx_tickers = [s["ticker"] for s in krx_summaries]

    hkex_bubble_html = build_bubble_chart(hkex_summaries)
    krx_bubble_html = build_bubble_chart(krx_summaries)

    hkex_sources_html = build_sources_chart(posts, hkex_tickers)
    krx_sources_html = build_sources_chart(posts, krx_tickers)

    predictions_chart_html = build_predictions_chart(predictions_log)

    # 5. Calculate run times (UTC and IST)
    now_utc = datetime.now(timezone.utc)
    now_ist = now_utc.astimezone(timezone(timedelta(hours=5, minutes=30)))
    run_time_utc = now_utc.strftime("%Y-%m-%d %H:%M UTC")
    run_time_ist = now_ist.strftime("%Y-%m-%d %H:%M IST")

    # 6. Extract Plotly standalone JS for offline embedding
    plotly_inline_js = f"<script>{po.get_plotlyjs()}</script>"

    # 7. Render Jinja2 Template
    templates_dir = Path(__file__).resolve().parent / "templates"
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(templates_dir)),
        autoescape=jinja2.select_autoescape(["html", "xml"]),
    )
    env.filters["format_price"] = format_price
    env.filters["format_int"] = format_int

    template = env.get_template("base.html")
    rendered_html = template.render(
        run_time_utc=run_time_utc,
        run_time_ist=run_time_ist,
        hkex_summaries=hkex_summaries,
        krx_summaries=krx_summaries,
        hkex_bubble_chart_html=hkex_bubble_html,
        krx_bubble_chart_html=krx_bubble_html,
        hkex_sources_chart_html=hkex_sources_html,
        krx_sources_chart_html=krx_sources_html,
        metrics=metrics,
        predictions_chart_html=predictions_chart_html,
        plotly_inline_js=plotly_inline_js,
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(rendered_html)

    print(f"Generated standalone offline report at {output_html} ({len(rendered_html)} bytes)")

    # 8. Export Leaderboard Image
    export_leaderboard_image(summary, output_img)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CrowdPulse offline HTML report and image")
    parser.add_argument("--run", type=str, default="data/outputs/sample", help="Path to input run directory")
    parser.add_argument("--output", type=str, default="report.html", help="Path to output report HTML")
    parser.add_argument("--img", type=str, default="leaderboard.png", help="Path to output leaderboard image")
    args = parser.parse_args()

    run_p = Path(args.run)
    out_html = Path(args.output)
    out_img = Path(args.img)

    build_report(run_p, out_html, out_img)


if __name__ == "__main__":
    main()
