"""Plotly chart generation module for CrowdPulse report.

Produces interactive, responsive Plotly charts with a dark financial theme.
Supports inline rendering without external CDN dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots

THEME = {
    "bg": "rgba(0,0,0,0)",
    "card_bg": "#111827",
    "grid_color": "#1f293d",
    "text_color": "#f8fafc",
    "muted_text": "#94a3b8",
    "bullish": "#10b981",
    "bearish": "#f43f5e",
    "neutral": "#94a3b8",
    "accent": "#3b82f6",
}


def build_bubble_chart(summaries: list[dict[str, Any]]) -> str:
    """Builds a Plotly scatter bubble chart: Sentiment (x) vs Mentions (y)."""
    fig = go.Figure()

    color_map = {
        "bullish": THEME["bullish"],
        "bearish": THEME["bearish"],
        "neutral": THEME["neutral"],
    }

    for dir_name in ["bullish", "neutral", "bearish"]:
        items = [s for s in summaries if s.get("direction") == dir_name]
        if not items:
            continue

        x_vals = [s.get("sentiment_score", 0.0) for s in items]
        y_vals = [s.get("mentions_10d", 0) for s in items]
        names = [f"{s.get('name')} ({s.get('ticker')})" for s in items]
        # Size based on mentions with minimum radius
        sizes = [max(18, min(55, int(s.get("mentions_10d", 100) / 4))) for s in items]

        custom_data = [
            (
                s.get("ticker"),
                s.get("name"),
                s.get("current_price"),
                s.get("currency"),
                s.get("mentions_10d"),
                s.get("sentiment_score"),
                s.get("confidence"),
                s.get("direction"),
            )
            for s in items
        ]

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers+text",
                name=dir_name.capitalize(),
                text=[s.get("ticker") for s in items],
                textposition="top center",
                textfont=dict(color=THEME["text_color"], size=11, family="ui-monospace, monospace"),
                marker=dict(
                    size=sizes,
                    color=color_map[dir_name],
                    opacity=0.85,
                    line=dict(width=1.5, color="#ffffff"),
                ),
                customdata=custom_data,
                hovertemplate=(
                    "<b>%{customdata[1]}</b> (%{customdata[0]})<br>"
                    "Price: %{customdata[2]} %{customdata[3]}<br>"
                    "10d Mentions: %{customdata[4]}<br>"
                    "Sentiment Score: %{customdata[5]}<br>"
                    "Direction: %{customdata[7]} (Confidence: %{customdata[6]:.0%})<br>"
                    "<extra></extra>"
                ),
            )
        )

    # Add reference quadrants
    fig.add_vline(x=0.15, line_width=1, line_dash="dash", line_color=THEME["grid_color"])
    fig.add_vline(x=-0.15, line_width=1, line_dash="dash", line_color=THEME["grid_color"])
    fig.add_vline(x=0.0, line_width=1, line_color=THEME["grid_color"])

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=THEME["bg"],
        plot_bgcolor=THEME["bg"],
        font=dict(color=THEME["text_color"], family="-apple-system, sans-serif"),
        margin=dict(l=40, r=40, t=30, b=40),
        height=320,
        xaxis=dict(
            title="Sentiment Score (-1.0 Bearish to +1.0 Bullish)",
            range=[-1.0, 1.0],
            zeroline=True,
            zerolinecolor=THEME["grid_color"],
            gridcolor=THEME["grid_color"],
        ),
        yaxis=dict(
            title="10-Day Total Mentions",
            gridcolor=THEME["grid_color"],
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    return fig.to_html(include_plotlyjs=False, full_html=False)


def build_candlestick_chart(ticker: str, prices: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    """Builds a 30-day candlestick chart with Support/Resistance and n+1 prediction."""
    # Filter for ticker and sort by date
    t_prices = [p for p in prices if p.get("ticker") == ticker]
    t_prices = sorted(t_prices, key=lambda x: x.get("date", ""))[-30:]

    if not t_prices:
        return "<div class='empty-state'>Price history unavailable</div>"

    dates = [p["date"] for p in t_prices]
    opens = [float(p["open"]) for p in t_prices]
    highs = [float(p["high"]) for p in t_prices]
    lows = [float(p["low"]) for p in t_prices]
    closes = [float(p["close"]) for p in t_prices]
    currency = summary.get("currency", "")

    fig = go.Figure()

    # Candlestick Trace
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name="Price",
            increasing_line_color=THEME["bullish"],
            decreasing_line_color=THEME["bearish"],
        )
    )

    # Support Lines (Green dashed)
    supports = summary.get("support", [])
    if len(supports) >= 1:
        s1 = float(supports[0])
        fig.add_hline(
            y=s1,
            line_dash="dash",
            line_color=THEME["bullish"],
            line_width=1.5,
            annotation_text=f"S1: {s1} {currency}",
            annotation_position="bottom right",
            annotation_font_color=THEME["bullish"],
            annotation_font_size=10,
        )
    if len(supports) >= 2:
        s2 = float(supports[1])
        fig.add_hline(
            y=s2,
            line_dash="dot",
            line_color=THEME["bullish"],
            line_width=1,
            annotation_text=f"S2: {s2} {currency}",
            annotation_position="bottom right",
            annotation_font_color=THEME["bullish"],
            annotation_font_size=9,
        )

    # Resistance Lines (Red dashed)
    resistances = summary.get("resistance", [])
    if len(resistances) >= 1:
        r1 = float(resistances[0])
        fig.add_hline(
            y=r1,
            line_dash="dash",
            line_color=THEME["bearish"],
            line_width=1.5,
            annotation_text=f"R1: {r1} {currency}",
            annotation_position="top right",
            annotation_font_color=THEME["bearish"],
            annotation_font_size=10,
        )
    if len(resistances) >= 2:
        r2 = float(resistances[1])
        fig.add_hline(
            y=r2,
            line_dash="dot",
            line_color=THEME["bearish"],
            line_width=1,
            annotation_text=f"R2: {r2} {currency}",
            annotation_position="top right",
            annotation_font_color=THEME["bearish"],
            annotation_font_size=9,
        )

    # Next-Day Prediction (n+1) Point & Whisker Range
    pred = summary.get("prediction", {})
    target_dt = pred.get("target_date")
    pred_close = pred.get("predicted_close")
    p_low = pred.get("low")
    p_high = pred.get("high")

    if target_dt and pred_close is not None:
        pred_close = float(pred_close)
        error_plus = float(p_high) - pred_close if p_high is not None else 0
        error_minus = pred_close - float(p_low) if p_low is not None else 0

        fig.add_trace(
            go.Scatter(
                x=[target_dt],
                y=[pred_close],
                mode="markers+text",
                name="Pred n+1 Close",
                text=[f"Target: {pred_close}"],
                textposition="top center",
                textfont=dict(color="#fde047", size=10, family="ui-monospace"),
                marker=dict(symbol="star", size=12, color="#fde047", line=dict(color="#000", width=1)),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[error_plus],
                    arrayminus=[error_minus],
                    color="#fde047",
                    thickness=2,
                    width=6,
                ),
                hovertemplate=(
                    f"<b>Predicted Close (n+1)</b>: {pred_close} {currency}<br>"
                    f"Range: {p_low} - {p_high} {currency}<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=THEME["bg"],
        plot_bgcolor=THEME["bg"],
        font=dict(color=THEME["text_color"], family="-apple-system, sans-serif"),
        margin=dict(l=40, r=40, t=20, b=30),
        height=260,
        xaxis=dict(
            rangeslider=dict(visible=False),
            gridcolor=THEME["grid_color"],
            type="category",
            showgrid=False,
        ),
        yaxis=dict(
            gridcolor=THEME["grid_color"],
            side="right",
            tickformat=",.2f" if currency == "HKD" else ",.0f",
        ),
        showlegend=False,
    )

    return fig.to_html(include_plotlyjs=False, full_html=False)


def build_volume_sentiment_chart(ticker: str, daily_rows: list[dict[str, Any]]) -> str:
    """Builds a dual-axis chart: Daily Mentions bars + Weighted Sentiment trendline."""
    t_daily = [d for d in daily_rows if d.get("ticker") == ticker]
    t_daily = sorted(t_daily, key=lambda x: x.get("date", ""))

    if not t_daily:
        return "<div class='empty-state'>Daily metrics unavailable</div>"

    dates = [d["date"] for d in t_daily]
    mentions = [int(d["mentions"]) for d in t_daily]
    sent_weighted = [float(d["social_sent_weighted"]) for d in t_daily]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Mentions Bar
    fig.add_trace(
        go.Bar(
            x=dates,
            y=mentions,
            name="Social Mentions",
            marker_color="rgba(59, 130, 246, 0.45)",
            marker_line=dict(color=THEME["accent"], width=1),
            hovertemplate="Date: %{x}<br>Mentions: %{y}<extra></extra>",
        ),
        secondary_y=False,
    )

    # Sentiment Line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=sent_weighted,
            name="Weighted Sentiment",
            mode="lines+markers",
            line=dict(color="#38bdf8", width=2.5),
            marker=dict(size=6, color="#0284c7"),
            hovertemplate="Date: %{x}<br>Weighted Sent: %{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.add_hline(
        y=0.0,
        line_dash="dot",
        line_color=THEME["grid_color"],
        secondary_y=True,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=THEME["bg"],
        plot_bgcolor=THEME["bg"],
        font=dict(color=THEME["text_color"], family="-apple-system, sans-serif"),
        margin=dict(l=35, r=35, t=10, b=25),
        height=180,
        xaxis=dict(showgrid=False, type="category"),
        yaxis=dict(title="Mentions", showgrid=False, zeroline=False),
        yaxis2=dict(
            title="Sentiment",
            range=[-1.0, 1.0],
            gridcolor=THEME["grid_color"],
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=9),
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    return fig.to_html(include_plotlyjs=False, full_html=False)


def build_sources_chart(posts_rows: list[dict[str, Any]], tickers: list[str]) -> str:
    """Builds a stacked bar chart of post counts per source per ticker."""
    # Find unique sources
    relevant_posts = [p for p in posts_rows if p.get("ticker") in tickers]
    sources = sorted(list({p.get("source", "") for p in relevant_posts if p.get("source")}))

    data_by_source: dict[str, list[int]] = {s: [] for s in sources}
    for t in tickers:
        t_posts = [p for p in relevant_posts if p.get("ticker") == t]
        for s in sources:
            cnt = sum(1 for p in t_posts if p.get("source") == s)
            data_by_source[s].append(cnt)

    fig = go.Figure()
    palette = ["#3b82f6", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6"]

    for idx, s in enumerate(sources):
        fig.add_trace(
            go.Bar(
                name=s.upper(),
                x=tickers,
                y=data_by_source[s],
                marker_color=palette[idx % len(palette)],
                hovertemplate="Ticker: %{x}<br>Source: " + s.upper() + "<br>Posts: %{y}<extra></extra>",
            )
        )

    fig.update_layout(
        barmode="stack",
        template="plotly_dark",
        paper_bgcolor=THEME["bg"],
        plot_bgcolor=THEME["bg"],
        font=dict(color=THEME["text_color"], family="-apple-system, sans-serif"),
        margin=dict(l=40, r=40, t=20, b=30),
        height=260,
        xaxis=dict(title="Ticker", gridcolor=THEME["grid_color"]),
        yaxis=dict(title="Post Volume", gridcolor=THEME["grid_color"]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    return fig.to_html(include_plotlyjs=False, full_html=False)


def build_predictions_chart(pred_log_rows: list[dict[str, Any]]) -> str:
    """Builds a comparison chart of predicted vs actual percentage return/error."""
    if not pred_log_rows:
        return "<div class='empty-state'>No historical predictions logged</div>"

    # Sort by target_date
    sorted_rows = sorted(pred_log_rows, key=lambda x: (x.get("target_date", ""), x.get("ticker", "")))

    fig = go.Figure()

    # Scatter plot of error percentage
    tickers = list({r["ticker"] for r in sorted_rows})
    for t in sorted(tickers):
        t_rows = [r for r in sorted_rows if r["ticker"] == t]
        fig.add_trace(
            go.Scatter(
                x=[r["target_date"] for r in t_rows],
                y=[float(r["error_pct"]) * 100 for r in t_rows],
                mode="lines+markers",
                name=t,
                hovertemplate="Ticker: " + t + "<br>Target Date: %{x}<br>Error: %{y:.2f}%<extra></extra>",
            )
        )

    fig.add_hline(
        y=0.0,
        line_color=THEME["grid_color"],
        line_width=1.5,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=THEME["bg"],
        plot_bgcolor=THEME["bg"],
        font=dict(color=THEME["text_color"], family="-apple-system, sans-serif"),
        margin=dict(l=40, r=40, t=25, b=30),
        height=280,
        xaxis=dict(title="Prediction Target Date", gridcolor=THEME["grid_color"]),
        yaxis=dict(title="Forecast Error (% Delta)", gridcolor=THEME["grid_color"]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10),
        ),
    )

    return fig.to_html(include_plotlyjs=False, full_html=False)


def export_leaderboard_image(summaries: list[dict[str, Any]], output_path: Path) -> None:
    """Exports a clean high-resolution leaderboard image as leaderboard.png using Kaleido."""
    # Build a clean summary table
    sorted_s = sorted(summaries, key=lambda x: int(x.get("mentions_10d", 0)), reverse=True)

    tickers = [s["ticker"] for s in sorted_s]
    names = [s["name"] for s in sorted_s]
    exchanges = [s["exchange"] for s in sorted_s]
    prices = [f"{s['current_price']} {s['currency']}" for s in sorted_s]
    mentions = [str(s["mentions_10d"]) for s in sorted_s]
    sentiments = [f"{s['sentiment_score']:+.2f}" for s in sorted_s]
    directions = [s["direction"].upper() for s in sorted_s]
    positions = [s["position"]["side"].upper() for s in sorted_s]
    preds = [f"{s['prediction']['predicted_close']} {s['currency']}" if s['prediction']['predicted_close'] else "N/A" for s in sorted_s]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[
                        "<b>EXCHANGE</b>",
                        "<b>TICKER</b>",
                        "<b>NAME</b>",
                        "<b>PRICE</b>",
                        "<b>10D MENTIONS</b>",
                        "<b>SENTIMENT</b>",
                        "<b>DIRECTION</b>",
                        "<b>POSITION</b>",
                        "<b>PRED CLOSE</b>",
                    ],
                    fill_color="#1e293b",
                    align="left",
                    font=dict(color="#ffffff", size=12, family="sans-serif"),
                    height=36,
                ),
                cells=dict(
                    values=[
                        exchanges,
                        tickers,
                        names,
                        prices,
                        mentions,
                        sentiments,
                        directions,
                        positions,
                        preds,
                    ],
                    fill_color="#111827",
                    align="left",
                    font=dict(color="#f8fafc", size=11, family="sans-serif"),
                    height=32,
                ),
            )
        ]
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0f19",
        margin=dict(l=20, r=20, t=30, b=20),
        width=1100,
        height=480,
        title=dict(
            text="CrowdPulse — Multi-Exchange Retail Sentiment & Predictive Rankings",
            font=dict(size=16, color="#ffffff"),
            x=0.02,
            y=0.96,
        ),
    )

    try:
        fig.write_image(str(output_path), scale=2)
        print(f"Exported leaderboard image to {output_path}")
    except Exception as e:
        print(f"Warning: Kaleido image export failed ({e}). Proceeding without breaking report generation.")
