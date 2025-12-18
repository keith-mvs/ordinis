"""
Ordinis Control Room - Dashboard.

Run with: streamlit run src/ordinis/dashboard/app.py
"""

from datetime import datetime
import json
from pathlib import Path
import random

import pandas as pd
import streamlit as st

# Page Config
st.set_page_config(
    page_title="Ordinis Control Room",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Compact dark theme inspired by modern trading terminals
st.markdown(
    """
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; font-family: 'Inter', sans-serif; }
    section[data-testid="stSidebar"] { background-color: #010409; border-right: 1px solid #30363d; }
    h1, h2, h3 { color: #f0f6fc !important; font-weight: 600; letter-spacing: -0.3px; }
    h1 { font-size: 1.8rem !important; margin-bottom: 0.35rem !important; }
    h2 { font-size: 1.35rem !important; margin-top: 0.75rem !important; }
    h3 { font-size: 1.05rem !important; margin-top: 0.35rem !important; }

    /* Metrics */
    div[data-testid="stMetric"] { background-color: #161b22; padding: 10px 14px; border-radius: 4px; border: 1px solid #30363d; box-shadow: none; }
    div[data-testid="stMetric"] label { color: #8b949e !important; font-size: 0.82rem !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f0f6fc !important; font-size: 1.2rem !important; }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

    /* Panels */
    .panel { background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 14px; margin-bottom: 14px; }

    /* Log stream */
    .log-entry { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; padding: 8px 0; border-bottom: 1px solid #21262d; display: flex; align-items: flex-start; }
    .log-time { color: #8b949e; min-width: 74px; margin-right: 10px; }
    .log-content { color: #c9d1d9; }
    .log-tag { display: inline-block; padding: 2px 7px; border-radius: 10px; font-size: 0.7rem; font-weight: 700; margin-right: 8px; text-transform: uppercase; }
    .tag-trade { background: rgba(46,160,67,0.12); color: #3fb950; border: 1px solid rgba(46,160,67,0.4); }
    .tag-ai { background: rgba(88,166,255,0.12); color: #58a6ff; border: 1px solid rgba(88,166,255,0.4); }
    .tag-alert { background: rgba(210,153,34,0.12); color: #d29922; border: 1px solid rgba(210,153,34,0.4); }

    /* Status pills */
    .status-dot { height: 8px; width: 8px; background: #3fb950; border-radius: 50%; display: inline-block; margin-right: 6px; }
    .status-text { color: #8b949e; font-size: 0.9rem; font-weight: 500; }

    /* Tables */
    div[data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 6px; }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
# Helpers
# -------------------------


def load_traces(file_path: Path) -> list:
    data = []
    if not file_path or not file_path.exists():
        return data
    with open(file_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data


def mock_series(
    days: int = 60, base: float = 100_000, drift: float = 250, noise: int = 900
) -> pd.DataFrame:
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days)
    values = [base + i * drift + random.randint(-noise, noise) for i in range(days)]
    return pd.DataFrame({"equity": values}, index=dates)


def mock_log_events() -> list:
    return [
        {"time": "10:42:15", "type": "trade", "msg": "Executed BUY 10 AAPL @ 150.20"},
        {
            "time": "10:42:01",
            "type": "ai",
            "msg": "Cortex: breakout pattern in AAPL (confidence 0.92)",
        },
        {
            "time": "10:41:55",
            "type": "ai",
            "msg": "Synapse: pulled 3 news snippets on Apple supply chain",
        },
        {"time": "10:30:00", "type": "alert", "msg": "Risk check: portfolio beta trimmed to 1.1"},
        {"time": "10:15:22", "type": "ai", "msg": "Cortex: re-evaluating tech exposure after CPI"},
        {"time": "09:30:01", "type": "trade", "msg": "Market open: execution engine live"},
    ]


# -------------------------
# Sidebar
# -------------------------


def render_sidebar() -> str:
    st.sidebar.markdown("## ORDINIS")
    st.sidebar.markdown(
        '<div style="margin-bottom: 18px; color: #8b949e; font-size: 0.82rem;">AI QUANT PLATFORM</div>',
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Live Ops", "AI System", "Benchmarks", "Strategy Lab", "System Health"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")
    st.sidebar.markdown(
        """
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <div class="status-dot"></div>
            <span style="color: #e0e0e0; font-size: 0.9rem;">Engine active</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.info("Market Regime: Volatile Bull")
    st.sidebar.text(f"Last update: {datetime.now().strftime('%H:%M:%S')}")
    return page


# -------------------------
# Overview (Home)
# -------------------------


def render_overview():
    header_left, header_right = st.columns([3, 1])
    with header_left:
        st.title("Dashboard")
        st.caption("Live portfolio, AI state, and market posture")
    with header_right:
        st.markdown(
            '<div style="text-align: right; padding-top: 8px;"><span class="status-text">LIVE TRADING</span></div>',
            unsafe_allow_html=True,
        )

    # Primary KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Equity", "$124,592", "+2.4%")
    k2.metric("Daily P&L", "$1,240", "+1.1%")
    k3.metric("Win Rate (7d)", "62%", "+4pp")
    k4.metric("Max Drawdown (30d)", "-3.8%", "Contained")

    # Secondary KPIs
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Active Strategies", "3", "2 live / 1 paper")
    s2.metric("Risk Budget Used", "54%", "Within limits")
    s3.metric("AI Confidence", "0.87", "High conviction")
    s4.metric("Market Regime", "Volatile Bull", "↑ momentum")

    st.markdown("---")

    main, side = st.columns([2, 1])

    # Performance and exposure
    with main:
        st.markdown("### Performance")
        perf_df = mock_series()
        st.line_chart(perf_df, height=260, color="#58a6ff")

        st.markdown("### Exposure by Sector")
        sector_df = pd.DataFrame(
            {
                "Sector": ["Tech", "Energy", "Financials", "Healthcare", "Discretionary"],
                "Exposure": [42, 18, 12, 9, 7],
            }
        ).set_index("Sector")
        st.bar_chart(sector_df, height=200)

        st.markdown("### Active Positions")
        positions = pd.DataFrame(
            {
                "Symbol": ["AAPL", "MSFT", "NVDA", "TSLA"],
                "Side": ["LONG", "LONG", "LONG", "SHORT"],
                "Size": [150, 80, 45, 200],
                "Entry": [145.00, 310.50, 450.00, 240.00],
                "Mark": [150.20, 315.00, 465.00, 235.50],
                "P&L": [780.00, 360.00, 675.00, 900.00],
                "AI Rationale": ["Momentum", "Earnings strength", "AI trend", "Overbought fade"],
            }
        )
        st.dataframe(
            positions,
            width="stretch",
            hide_index=True,
            column_config={
                "P&L": st.column_config.NumberColumn(format="$%.2f"),
                "Entry": st.column_config.NumberColumn(format="$%.2f"),
                "Mark": st.column_config.NumberColumn(format="$%.2f"),
            },
        )

        st.markdown("### Open Orders")
        orders = pd.DataFrame(
            {
                "Symbol": ["AMZN", "META"],
                "Side": ["BUY", "SELL"],
                "Qty": [50, 30],
                "Limit": [170.5, 355.0],
                "Status": ["Working", "Working"],
            }
        )
        st.dataframe(orders, width="stretch", hide_index=True)

    # Right rail: AI state + activity
    with side:
        st.markdown("### AI State")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("Engine: Stable")
        st.markdown("Cortex latency: 420 ms")
        st.markdown("Tokens (today): 45,200")
        st.markdown("Guardrails: Enabled")
        st.markdown("Top context source: Tech earnings transcripts")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### Neural Stream")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        for evt in mock_log_events():
            tag_class = f"tag-{evt['type']}"
            st.markdown(
                f"""
                <div class="log-entry">
                    <div class="log-time">{evt['time']}</div>
                    <div class="log-content">
                        <span class="log-tag {tag_class}">{evt['type']}</span>
                        {evt['msg']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### Watchlist")
        watchlist = pd.DataFrame(
            {
                "Ticker": ["AMD", "INTC", "QCOM"],
                "Signal": ["Bullish", "Neutral", "Bearish"],
                "Score": [0.85, 0.50, 0.32],
            }
        )
        st.dataframe(watchlist, width="stretch", hide_index=True)


# -------------------------
# Live Ops (Traces)
# -------------------------


def render_live_ops():
    st.title("Live Operations and Traces")

    log_dir = Path("artifacts/traces")
    if not log_dir.exists():
        st.error(f"Log directory not found: {log_dir}")
        return

    log_files = sorted(log_dir.glob("trace_*.jsonl"), reverse=True)
    if not log_files:
        st.warning("No trace logs found.")
        return

    col_file, col_info = st.columns([3, 1])
    with col_file:
        selected_file = st.selectbox("Select trace log", log_files, format_func=lambda p: p.name)
    with col_info:
        st.caption("Filters help trim noise for targeted debugging.")

    traces = load_traces(selected_file)
    df = pd.DataFrame(traces)
    if df.empty:
        st.warning("Selected log file is empty.")
        return

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        df["time_str"] = df["timestamp"].dt.strftime("%H:%M:%S")

    types = df["type"].unique().tolist() if "type" in df.columns else []
    components = df["component"].unique().tolist() if "component" in df.columns else []

    f1, f2 = st.columns(2)
    with f1:
        selected_types = st.multiselect("Trace type", options=types, default=types)
    with f2:
        selected_components = st.multiselect("Component", options=components, default=components)

    filtered = df.copy()
    if "type" in filtered.columns:
        filtered = filtered[filtered["type"].isin(selected_types)]
    if "component" in filtered.columns:
        filtered = filtered[filtered["component"].isin(selected_components)]

    st.markdown(f"### Event log ({len(filtered)} events)")

    for _, row in filtered.iterrows():
        trace_id = row.get("trace_id", "N/A")
        component = row.get("component", "Unknown")
        type_ = row.get("type", "Unknown")
        time_str = row.get("time_str", "")

        with st.expander(f"[{time_str}] {component} / {type_}"):
            st.text(f"Trace ID: {trace_id}")
            content = row.get("content", {})

            if type_ == "llm_request":
                st.json(content)
            elif type_ == "llm_response":
                text_content = (
                    content.get("content", "") if isinstance(content, dict) else str(content)
                )
                if "<think>" in text_content:
                    parts = text_content.split("</think>")
                    if len(parts) > 1:
                        st.info(parts[0].replace("<think>", "").strip())
                        st.success(parts[1].strip())
                    else:
                        st.code(text_content)
                else:
                    st.code(text_content)
                if isinstance(content, dict) and "usage" in content:
                    st.json({"usage": content.get("usage")})
            else:
                st.json(content)

            if row.get("metadata"):
                st.markdown("Metadata")
                st.json(row["metadata"])


# -------------------------
# AI System (state + lineage)
# -------------------------


def render_ai_system():
    st.title("AI System State")
    st.caption("Models, retrieval, and guardrail telemetry")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cortex latency (p95)", "420 ms", "-35 ms")
    m2.metric("Tokens today", "45,200", "+5%")
    m3.metric("RAG freshness", "12 min", "Within SLA")
    m4.metric("Guardrail blocks", "2", "Past 24h")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Model Versions")
        st.markdown(
            """
            - Cortex: nemotron-ultra (prod)
            - Judge: nemotron-reward (nightly eval)
            - Embeddings: e5-large (v2)
            - Safety: moderation-v1
            """
        )

    with c2:
        st.markdown("### Retrieval Snapshot")
        st.markdown(
            """
            - Top collections: earnings, macro, factor-notes
            - Index size: 4.2M chunks
            - Mean hitrate (24h): 82%
            - Avg latency: 110 ms
            """
        )

    st.markdown("### Prompt Health (sample)")
    prompt_df = pd.DataFrame(
        {
            "Prompt": ["Cortex/system", "Guardrails", "Judge"],
            "Last update": ["2025-12-10", "2025-12-12", "2025-12-09"],
            "Version": ["v23", "v11", "v7"],
            "Notes": [
                "Added macro bias check",
                "Tightened risk phrasing",
                "Aligned reward weights",
            ],
        }
    )
    st.dataframe(prompt_df, width="stretch", hide_index=True)


# -------------------------
# Benchmarks
# -------------------------


def render_benchmarks():
    st.title("Benchmarks & Model Performance")
    st.caption("Comparative analysis of backend model inference and quality")

    # Top-level metrics
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Avg Latency (All)", "385 ms", "-12 ms")
    b2.metric("Avg Tokens/Sec", "42.5", "+1.2")
    b3.metric("Total Cost (24h)", "$12.45", "+$2.10")
    b4.metric("Judge Score (Avg)", "8.9/10", "+0.2")

    st.markdown("---")

    # Charts Section
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Latency by Model (ms)")
        latency_data = pd.DataFrame(
            {
                "Model": ["nemotron-ultra", "gpt-4-turbo", "claude-3-opus", "llama-3-70b"],
                "Latency": [420, 650, 980, 210],
            }
        ).set_index("Model")
        st.bar_chart(latency_data, color="#58a6ff")

    with c2:
        st.markdown("### Success Rate (Judge Eval)")
        success_data = pd.DataFrame(
            {
                "Model": ["nemotron-ultra", "gpt-4-turbo", "claude-3-opus", "llama-3-70b"],
                "Success Rate": [0.92, 0.95, 0.94, 0.85],
            }
        ).set_index("Model")
        st.bar_chart(success_data, color="#3fb950")

    # Detailed Table
    st.markdown("### Model Leaderboard")
    leaderboard = pd.DataFrame(
        {
            "Model": ["nemotron-ultra", "gpt-4-turbo", "claude-3-opus", "llama-3-70b"],
            "Role": ["Primary Reasoning", "Fallback", "Creative/Drafting", "Fast Classification"],
            "Cost ($/1M)": [10.00, 30.00, 15.00, 0.70],
            "Throughput (t/s)": [45, 22, 18, 85],
            "Error Rate": ["0.05%", "0.01%", "0.02%", "0.12%"],
        }
    )
    st.dataframe(
        leaderboard,
        width="stretch",
        hide_index=True,
        column_config={
            "Cost ($/1M)": st.column_config.NumberColumn(format="$%.2f"),
        },
    )

    st.markdown("### Recent Benchmark Runs")
    runs = pd.DataFrame(
        {
            "Run ID": ["BM-2025-001", "BM-2025-002", "BM-2025-003"],
            "Date": ["2025-12-14 10:00", "2025-12-14 09:00", "2025-12-13 22:00"],
            "Dataset": ["Financial QA v2", "Code Gen v1", "Reasoning Hard"],
            "Models": ["nemotron-ultra", "gpt-4-turbo", "claude-3-opus"],
            "Status": ["Completed", "Completed", "Completed"],
        }
    )
    st.dataframe(runs, width="stretch", hide_index=True)


# -------------------------
# Strategy Lab
# -------------------------


def render_strategy_lab():
    st.title("Strategy Lab")
    st.caption("Ideation, what-if replays, and backtesting")

    st.markdown("### Counterfactual Sandbox")
    st.info("Replay a historical event with modified prompts, models, or context. (Coming soon)")

    st.markdown("### Strategy Backlog")
    strategies = pd.DataFrame(
        {
            "Name": ["Tech Momentum Alpha", "Volatility Harvest", "Sentiment Arb"],
            "Status": ["Live", "Live", "Paper"],
            "Sharpe": [2.4, 1.8, 1.2],
            "Drawdown": ["-5.2%", "-8.1%", "-3.4%"],
            "YTD": ["+24.5%", "+12.2%", "+4.5%"],
        }
    )
    st.dataframe(strategies, width="stretch", hide_index=True)


# -------------------------
# System Health
# -------------------------


def render_system_health():
    st.title("System Health")
    st.caption("Infrastructure and service telemetry")

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("API latency", "124 ms", "-12 ms")
    t2.metric("Throughput (rps)", "72", "+5")
    t3.metric("Error rate", "0.01%", "Stable")
    t4.metric("Est. cost (today)", "$4.20", "+$0.50")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Compute Load")
        st.progress(45, text="CPU 45%")
        st.progress(62, text="Memory 62%")
        st.progress(28, text="GPU 28%")

    with col_b:
        st.markdown("### Services")
        services = [
            {"name": "Market data feed", "status": "Operational", "color": "#3fb950"},
            {"name": "Execution gateway", "status": "Operational", "color": "#3fb950"},
            {"name": "LLM inference (Cortex)", "status": "Operational", "color": "#3fb950"},
            {"name": "Vector DB (Synapse)", "status": "Degraded", "color": "#d29922"},
        ]
        for svc in services:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #21262d;">
                    <span>{svc['name']}</span>
                    <span style="color: {svc['color']}; font-weight: 700;">{svc['status']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


# -------------------------
# Main
# -------------------------


def main():
    page = render_sidebar()
    if page == "Overview":
        render_overview()
    elif page == "Live Ops":
        render_live_ops()
    elif page == "AI System":
        render_ai_system()
    elif page == "Strategy Lab":
        render_strategy_lab()
    elif page == "System Health":
        render_system_health()


if __name__ == "__main__":
    main()
