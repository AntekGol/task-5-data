import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from data_loader import DEFAULT_SHEET_URL, fetch_csv_text, parse_simulator_csv
from analytics import analyze_all, fit_polynomial_trend
from pdf_report import generate_pdf_report

st.set_page_config(
    page_title="Weyland-Yutani Mining Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
    .wy-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        padding: 1.25rem 1.6rem;
        border-radius: 10px;
        border-left: 6px solid #f59e0b;
        margin-bottom: 1.2rem;
    }
    .wy-title { font-size: 1.55rem; font-weight: 800; letter-spacing: 0.04em; margin: 0; }
    .wy-sub { font-size: 0.92rem; color: #cbd5e1; margin-top: 0.25rem; }
    .anom-card {
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.75rem;
        border: 1px solid #e2e8f0;
        background: #ffffff;
    }
    .anom-spike { border-left: 5px solid #dc2626; background: #fff5f5; }
    .anom-drop { border-left: 5px solid #7c3aed; background: #f5f3ff; }
    .badge-spike {
        background: #dc2626; color: white; padding: 2px 8px;
        border-radius: 4px; font-size: 0.78rem; font-weight: 700;
    }
    .badge-drop {
        background: #7c3aed; color: white; padding: 2px 8px;
        border-radius: 4px; font-size: 0.78rem; font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

COLORS = ["#2563eb", "#d97706", "#059669", "#7c3aed", "#0891b2", "#db2777", "#334155"]


def build_plotly_chart(df, series_list, chart_type, poly_degree, series_details, show_trend=True, show_outliers=True):
    fig = go.Figure()
    dates = df["Date"]

    for i, s_name in enumerate(series_list):
        color = COLORS[i % len(COLORS)]
        vals = df[s_name].to_numpy(dtype=float)

        if chart_type == "Line":
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=vals,
                    mode="lines+markers",
                    name=s_name,
                    line=dict(color=color, width=2.4),
                    marker=dict(size=5),
                )
            )
        elif chart_type == "Bar":
            fig.add_trace(
                go.Bar(
                    x=dates,
                    y=vals,
                    name=s_name,
                    marker_color=color,
                    opacity=0.86,
                )
            )
        elif chart_type == "Stacked":
            fig.add_trace(
                go.Bar(
                    x=dates,
                    y=vals,
                    name=s_name,
                    marker_color=color,
                    opacity=0.88,
                )
            )

        if show_trend and chart_type != "Stacked":
            trend_vals = fit_polynomial_trend(vals, poly_degree)
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=trend_vals,
                    mode="lines",
                    name=f"{s_name} Trend (deg {poly_degree})",
                    line=dict(color=color, width=2, dash="dash"),
                )
            )

        if show_outliers and chart_type != "Stacked":
            anoms = series_details.get(s_name, {}).get("filtered_anomalies", [])
            spikes = [a for a in anoms if a["type"] == "Spike"]
            drops = [a for a in anoms if a["type"] == "Drop"]
            if spikes:
                fig.add_trace(
                    go.Scatter(
                        x=[dates.iloc[a["index"]] for a in spikes],
                        y=[a["value"] for a in spikes],
                        mode="markers",
                        name=f"{s_name} Spike",
                        marker=dict(color="#dc2626", size=12, symbol="triangle-up", line=dict(width=1.5, color="white")),
                        customdata=[[a["tests_str"], a["pct_from_ma"], a["z_score"]] for a in spikes],
                        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Spike: %{y:.2f}t<br>MA Diff: %{customdata[1]:+.1f}%<br>Z: %{customdata[2]:+.2f}<br>Tests: %{customdata[0]}<extra></extra>",
                    )
                )
            if drops:
                fig.add_trace(
                    go.Scatter(
                        x=[dates.iloc[a["index"]] for a in drops],
                        y=[a["value"] for a in drops],
                        mode="markers",
                        name=f"{s_name} Drop",
                        marker=dict(color="#7c3aed", size=12, symbol="triangle-down", line=dict(width=1.5, color="white")),
                        customdata=[[a["tests_str"], a["pct_from_ma"], a["z_score"]] for a in drops],
                        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Drop: %{y:.2f}t<br>MA Diff: %{customdata[1]:+.1f}%<br>Z: %{customdata[2]:+.2f}<br>Tests: %{customdata[0]}<extra></extra>",
                    )
                )

    if chart_type == "Stacked":
        tot_vals = df["Total Output"].to_numpy(dtype=float) if len(series_list) > 1 else df[series_list[0]].to_numpy(dtype=float)
        target_name = "Total Output" if len(series_list) > 1 else series_list[0]
        if show_trend:
            tot_trend = fit_polynomial_trend(tot_vals, poly_degree)
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=tot_trend,
                    mode="lines",
                    name=f"{target_name} Trend (deg {poly_degree})",
                    line=dict(color="#0f172a", width=2.8, dash="dash"),
                )
            )
        if show_outliers:
            tot_anoms = series_details.get(target_name, {}).get("filtered_anomalies", [])
            spikes = [a for a in tot_anoms if a["type"] == "Spike"]
            drops = [a for a in tot_anoms if a["type"] == "Drop"]
            if spikes:
                fig.add_trace(
                    go.Scatter(
                        x=[dates.iloc[a["index"]] for a in spikes],
                        y=[a["value"] for a in spikes],
                        mode="markers",
                        name=f"{target_name} Spike",
                        marker=dict(color="#dc2626", size=13, symbol="triangle-up", line=dict(width=1.5, color="white")),
                    )
                )
            if drops:
                fig.add_trace(
                    go.Scatter(
                        x=[dates.iloc[a["index"]] for a in drops],
                        y=[a["value"] for a in drops],
                        mode="markers",
                        name=f"{target_name} Drop",
                        marker=dict(color="#7c3aed", size=13, symbol="triangle-down", line=dict(width=1.5, color="white")),
                    )
                )

    barmode = "stack" if chart_type == "Stacked" else ("group" if chart_type == "Bar" else None)
    fig.update_layout(
        barmode=barmode,
        height=470,
        margin=dict(l=20, r=20, t=35, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title="Date",
        yaxis_title="Resource Extraction (tons)",
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


with st.sidebar:
    st.markdown("### Data Source")
    sheet_url = st.text_input("Google Spreadsheet URL or ID", value=DEFAULT_SHEET_URL)
    refresh_clicked = st.button("Refresh Data", use_container_width=True, type="primary")
    st.markdown(f"[Open Generator Spreadsheet]({sheet_url})")

    st.divider()
    st.markdown("### Chart Options")
    chart_type = st.selectbox("Chart Type", ["Line", "Bar", "Stacked"], index=0)
    poly_degree = st.selectbox("Polynomial Trendline Degree", [1, 2, 3, 4], index=2)
    show_trend = st.checkbox("Show Trendlines", value=True)
    show_outliers = st.checkbox("Highlight Outliers", value=True)

    st.divider()
    st.markdown("### Anomaly Tests")
    highlight_mode = st.selectbox(
        "Outlier Filter",
        ["Any Test (Union)", "Consensus (2+ Tests)", "IQR Rule", "Z-Score", "Moving Average (%)", "Grubbs' Test"],
        index=0,
    )
    iqr_k = st.slider("IQR Multiplier (k)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
    z_threshold = st.slider("Z-Score Threshold", min_value=1.0, max_value=3.5, value=2.0, step=0.1)
    ma_window = st.slider("Moving Average Window (days)", min_value=3, max_value=15, value=7, step=2)
    ma_pct = st.slider("Distance from Moving Avg (%)", min_value=10.0, max_value=80.0, value=45.0, step=2.5)
    grubbs_alpha = st.select_slider(
        "Grubbs Test Alpha",
        options=[0.01, 0.025, 0.05, 0.10, 0.15, 0.20],
        value=0.05,
    )

params = {
    "iqr_k": iqr_k,
    "z_threshold": z_threshold,
    "ma_window": ma_window,
    "ma_pct": ma_pct,
    "grubbs_alpha": grubbs_alpha,
}

if "cached_csv" not in st.session_state or refresh_clicked or st.session_state.get("last_url") != sheet_url:
    csv_text, source_label = fetch_csv_text(sheet_url)
    st.session_state["cached_csv"] = csv_text
    st.session_state["source_label"] = source_label
    st.session_state["last_url"] = sheet_url

csv_text = st.session_state["cached_csv"]
source_label = st.session_state["source_label"]
df, mine_names, meta = parse_simulator_csv(csv_text)
stats_df, anomalies, series_details = analyze_all(df, mine_names, params, highlight_mode=highlight_mode)

st.markdown(
    f"""
    <div class="wy-banner">
        <div class="wy-title">WEYLAND-YUTANI CORPORATION — MINING OPS ANALYTICS</div>
        <div class="wy-sub">
            Source: <b>{source_label}</b> &nbsp;|&nbsp;
            Active Mines: <b>{len(mine_names)} ({", ".join(mine_names)})</b> &nbsp;|&nbsp;
            Range: <b>{df['Date'].min().strftime('%Y-%m-%d')} → {df['Date'].max().strftime('%Y-%m-%d')} ({len(df)} days)</b> &nbsp;|&nbsp;
            Distribution: <b>{meta.get('distribution', 'Normal')}</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

top_col1, top_col2, top_col3, top_col4, top_col5 = st.columns([1.1, 1.1, 1.1, 1.1, 1.4])
tot_row = stats_df[stats_df["Mine / Series"] == "Total Output"].iloc[0]
spike_count = sum(1 for a in anomalies if a["type"] == "Spike")
drop_count = sum(1 for a in anomalies if a["type"] == "Drop")

top_col1.metric("Mean Total Daily Output", f"{tot_row['Mean']:.2f} t", f"Median: {tot_row['Median']:.2f} t")
top_col2.metric("Total Output Std Dev", f"{tot_row['Std Dev']:.2f} t", f"IQR: {tot_row['IQR']:.2f} t")
top_col3.metric("Detected Spikes", f"{spike_count}", "Positive Outliers")
top_col4.metric("Detected Drops", f"{drop_count}", "Negative Outliers", delta_color="inverse")

with top_col5:
    pdf_bytes = generate_pdf_report(
        df=df,
        mine_names=mine_names,
        stats_df=stats_df,
        anomalies=anomalies,
        series_details=series_details,
        params=params,
        chart_type=chart_type,
        poly_degree=poly_degree,
        highlight_mode=highlight_mode,
        meta=meta,
    )
    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name="Weyland_Yutani_Mining_Report.pdf",
        mime="application/pdf",
        use_container_width=True,
        type="primary",
    )

st.subheader("1. Statistical Summary — Each Mine & Total Output")
st.dataframe(
    stats_df.style.format({
        "Mean": "{:.2f}",
        "Std Dev": "{:.2f}",
        "Median": "{:.2f}",
        "IQR": "{:.2f}",
        "Min": "{:.2f}",
        "Max": "{:.2f}",
    }),
    use_container_width=True,
    hide_index=True,
)

st.subheader("2. Interactive Extraction Charts, Outliers & Polynomial Trendlines")
view_tabs = st.tabs(["All Mines Overview", "Total Output Analysis"] + [f"Mine: {m}" for m in mine_names])

with view_tabs[0]:
    fig_all = build_plotly_chart(df, mine_names, chart_type, poly_degree, series_details, show_trend, show_outliers)
    st.plotly_chart(fig_all, use_container_width=True)

with view_tabs[1]:
    fig_tot = build_plotly_chart(df, ["Total Output"], chart_type, poly_degree, series_details, show_trend, show_outliers)
    st.plotly_chart(fig_tot, use_container_width=True)

for idx, m_name in enumerate(mine_names):
    with view_tabs[idx + 2]:
        fig_m = build_plotly_chart(df, [m_name], chart_type, poly_degree, series_details, show_trend, show_outliers)
        st.plotly_chart(fig_m, use_container_width=True)

st.subheader(f"3. Detected Anomalies — Detailed Sections ({len(anomalies)} Events)")

filter_c1, filter_c2 = st.columns(2)
with filter_c1:
    selected_series_filter = st.multiselect(
        "Filter by Mine / Series",
        options=mine_names + ["Total Output"],
        default=mine_names + ["Total Output"],
    )
with filter_c2:
    selected_type_filter = st.multiselect(
        "Filter by Anomaly Type",
        options=["Spike", "Drop"],
        default=["Spike", "Drop"],
    )

visible_anomalies = [
    a for a in anomalies
    if a["series"] in selected_series_filter and a["type"] in selected_type_filter
]

if not visible_anomalies:
    st.info("No anomalies match the current filter criteria. Try lowering the test thresholds in the sidebar.")
else:
    anom_table = pd.DataFrame([
        {
            "Date": a["date"],
            "Mine / Series": a["series"],
            "Type": a["type"],
            "Output (t)": a["value"],
            "Moving Avg (t)": a["moving_avg"],
            "Diff from MA (%)": f"{a['pct_from_ma']:+.1f}%",
            "Series Mean (t)": a["mean"],
            "Z-Score": f"{a['z_score']:+.2f}",
            "Triggered Tests": a["tests_str"],
        }
        for a in visible_anomalies
    ])
    st.dataframe(anom_table, use_container_width=True, hide_index=True)

    st.markdown("#### Individual Anomaly Breakdown")
    cols = st.columns(2)
    for i, a in enumerate(visible_anomalies):
        is_spike = a["type"] == "Spike"
        card_cls = "anom-spike" if is_spike else "anom-drop"
        badge_cls = "badge-spike" if is_spike else "badge-drop"
        direction_word = "surged above" if is_spike else "dropped below"
        with cols[i % 2]:
            st.markdown(
                f"""
                <div class="anom-card {card_cls}">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <div>
                            <span class="{badge_cls}">{a['type'].upper()}</span>
                            <b style="margin-left:8px; font-size:1.02rem; color:#0f172a;">{a['series']} — {a['date']}</b>
                        </div>
                        <span style="font-size:0.85rem; color:#475569;"><b>{a['test_count']}/4</b> tests triggered</span>
                    </div>
                    <div style="font-size:0.9rem; color:#1e293b; margin-bottom:4px;">
                        <b>Recorded Output:</b> {a['value']:.2f} t &nbsp;|&nbsp;
                        <b>Moving Baseline:</b> {a['moving_avg']:.2f} t ({a['pct_from_ma']:+.1f}%) &nbsp;|&nbsp;
                        <b>Z-Score:</b> {a['z_score']:+.2f}
                    </div>
                    <div style="font-size:0.86rem; color:#334155; margin-bottom:4px;">
                        <b>Triggered Tests:</b> {a['tests_str']}
                    </div>
                    <div style="font-size:0.84rem; color:#475569;">
                        Daily extraction at <b>{a['series']}</b> {direction_word} the local moving average by
                        <b>{abs(a['pct_from_ma']):.1f}%</b> (series mean: {a['mean']:.2f} t).
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

with st.expander("Raw Data & Pivoted Table"):
    rec_df = meta.get("records_df")
    if rec_df is not None:
        c_rec, c_piv = st.columns([1, 1.3])
        with c_rec:
            st.markdown("**Normalized Source Records (`Date | Mine | Output`)**")
            st.dataframe(rec_df, use_container_width=True, hide_index=True)
        with c_piv:
            st.markdown("**Transposed / Pivoted Daily View (UI Presentation)**")
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)
