import os
import tempfile
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from fpdf import FPDF
from analytics import fit_polynomial_trend


PALETTE = ["#2563eb", "#d97706", "#059669", "#7c3aed", "#0891b2", "#db2777", "#334155"]


def render_overview_chart(df, mine_names, chart_type, poly_degree, series_details, out_path):
    fig, ax = plt.subplots(figsize=(10.5, 4.4), dpi=160)
    dates = df["Date"]
    x_num = np.arange(len(df))

    if chart_type == "Stacked":
        bottom = np.zeros(len(df))
        for i, m in enumerate(mine_names):
            vals = df[m].to_numpy(dtype=float)
            color = PALETTE[i % len(PALETTE)]
            ax.bar(dates, vals, bottom=bottom, label=m, color=color, alpha=0.82, width=0.72)
            bottom += vals
        tot_trend = fit_polynomial_trend(df["Total Output"].to_numpy(dtype=float), poly_degree)
        ax.plot(dates, tot_trend, color="#0f172a", linestyle="--", linewidth=2.2, label=f"Total Trend (deg {poly_degree})")
        tot_anoms = series_details.get("Total Output", {}).get("filtered_anomalies", [])
        spikes = [a for a in tot_anoms if a["type"] == "Spike"]
        drops = [a for a in tot_anoms if a["type"] == "Drop"]
        if spikes:
            ax.scatter([df["Date"].iloc[a["index"]] for a in spikes], [a["value"] for a in spikes], color="#dc2626", s=70, zorder=6, label="Spike Outlier")
        if drops:
            ax.scatter([df["Date"].iloc[a["index"]] for a in drops], [a["value"] for a in drops], color="#7c3aed", s=70, marker="v", zorder=6, label="Drop Outlier")
    elif chart_type == "Bar":
        n_m = max(1, len(mine_names))
        width = 0.78 / n_m
        for i, m in enumerate(mine_names):
            vals = df[m].to_numpy(dtype=float)
            color = PALETTE[i % len(PALETTE)]
            offset = (i - (n_m - 1) / 2.0) * width
            bar_dates = mdates.date2num(dates) + offset
            ax.bar(bar_dates, vals, width=width, label=m, color=color, alpha=0.85)
            trend = fit_polynomial_trend(vals, poly_degree)
            ax.plot(dates, trend, color=color, linestyle="--", linewidth=1.6)
            anoms = series_details.get(m, {}).get("filtered_anomalies", [])
            for a in anoms:
                mk = "^" if a["type"] == "Spike" else "v"
                mc = "#dc2626" if a["type"] == "Spike" else "#7c3aed"
                ax.scatter(df["Date"].iloc[a["index"]], a["value"], color=mc, marker=mk, s=55, zorder=6)
        ax.xaxis_date()
    else:
        for i, m in enumerate(mine_names):
            vals = df[m].to_numpy(dtype=float)
            color = PALETTE[i % len(PALETTE)]
            ax.plot(dates, vals, label=m, color=color, linewidth=1.9)
            trend = fit_polynomial_trend(vals, poly_degree)
            ax.plot(dates, trend, color=color, linestyle="--", linewidth=1.3, alpha=0.8)
            anoms = series_details.get(m, {}).get("filtered_anomalies", [])
            for a in anoms:
                mk = "o" if a["type"] == "Spike" else "v"
                mc = "#dc2626" if a["type"] == "Spike" else "#7c3aed"
                ax.scatter(df["Date"].iloc[a["index"]], a["value"], color=mc, marker=mk, s=55, edgecolors="white", linewidth=0.7, zorder=6)

    ax.set_title(f"Daily Mine Extraction ({chart_type} Chart) & Degree-{poly_degree} Polynomial Trend", fontsize=11, fontweight="bold", pad=10)
    ax.set_ylabel("Extraction (tons)", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8, ncol=3, frameon=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def render_single_series_chart(df, series_name, poly_degree, series_info, color, out_path):
    fig, ax = plt.subplots(figsize=(10.0, 3.2), dpi=150)
    dates = df["Date"]
    vals = df[series_name].to_numpy(dtype=float)
    ma = series_info["details"]["moving_avg"]
    trend = fit_polynomial_trend(vals, poly_degree)

    ax.plot(dates, vals, color=color, linewidth=1.8, label=f"{series_name} Output")
    ax.plot(dates, trend, color="#0f172a", linestyle="--", linewidth=1.5, label=f"Poly Trend (deg {poly_degree})")
    ax.plot(dates, ma, color="#64748b", linestyle=":", linewidth=1.3, label="Moving Baseline")

    anoms = series_info["filtered_anomalies"]
    spikes = [a for a in anoms if a["type"] == "Spike"]
    drops = [a for a in anoms if a["type"] == "Drop"]
    if spikes:
        ax.scatter([dates.iloc[a["index"]] for a in spikes], [a["value"] for a in spikes], color="#dc2626", s=65, zorder=6, label="Spike")
    if drops:
        ax.scatter([dates.iloc[a["index"]] for a in drops], [a["value"] for a in drops], color="#7c3aed", marker="v", s=65, zorder=6, label="Drop")

    ax.set_title(f"{series_name} — Extraction Profile, Trendline & Detected Outliers", fontsize=10, fontweight="bold")
    ax.set_ylabel("Tons", fontsize=8.5)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=7.5, ncol=4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


class WeylandReportPDF(FPDF):
    def header(self):
        self.set_fill_color(15, 23, 42)
        self.rect(0, 0, 210, 18, "F")
        self.set_y(5)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(248, 250, 252)
        self.cell(0, 8, "WEYLAND-YUTANI CORPORATION  |  MINING OPS TELEMETRY & ANOMALY REPORT", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 116, 139)
        self.cell(0, 8, f"Weyland-Yutani Resource Analytics Division  -  Page {self.page_no()}/{{nb}}", align="C")


def safe_text(val):
    return str(val).encode("latin-1", "replace").decode("latin-1")


def generate_pdf_report(df, mine_names, stats_df, anomalies, series_details, params, chart_type, poly_degree, highlight_mode, meta):
    pdf = WeylandReportPDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 9, "Executive Extraction & Anomaly Intelligence Report", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(71, 85, 105)
    d_start = df["Date"].min().strftime("%Y-%m-%d") if len(df) else "N/A"
    d_end = df["Date"].max().strftime("%Y-%m-%d") if len(df) else "N/A"
    gen_ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    pdf.cell(0, 5.5, safe_text(f"Observation Period: {d_start} to {d_end} ({len(df)} days)   |   Active Mines: {', '.join(mine_names)}   |   Generated: {gen_ts}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5.5, safe_text(f"Chart Mode: {chart_type}   |   Trendline: Polynomial Degree {poly_degree}   |   Filter Mode: {highlight_mode}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5.5, safe_text(f"Test Thresholds: IQR k={params['iqr_k']}  |  Z-Score={params['z_threshold']}  |  Moving Avg={params['ma_pct']}% (win={params['ma_window']}d)  |  Grubbs alpha={params['grubbs_alpha']}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "1. Descriptive Statistics (Per Mine & Total Output)", new_x="LMARGIN", new_y="NEXT")

    col_widths = [46, 24, 24, 24, 24, 24, 24]
    headers = ["Mine / Series", "Mean", "Std Dev", "Median", "IQR", "Min", "Max"]
    pdf.set_fill_color(30, 41, 59)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9)
    for w, h in zip(col_widths, headers):
        pdf.cell(w, 7, h, border=1, align="C", fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 9)
    for idx, row in stats_df.iterrows():
        is_total = row["Mine / Series"] == "Total Output"
        if is_total:
            pdf.set_fill_color(226, 232, 240)
            pdf.set_font("Helvetica", "B", 9)
        else:
            pdf.set_fill_color(248, 250, 252 if idx % 2 == 0 else 255)
            pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(col_widths[0], 6.8, safe_text(row["Mine / Series"]), border=1, align="L", fill=True)
        for w, col_name in zip(col_widths[1:], headers[1:]):
            pdf.cell(w, 6.8, f"{row[col_name]:.2f}", border=1, align="R", fill=True)
        pdf.ln()

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "2. Extraction Output Charts & Polynomial Trendlines", new_x="LMARGIN", new_y="NEXT")

    temp_files = []
    try:
        fd_main, main_chart_path = tempfile.mkstemp(suffix=".png")
        os.close(fd_main)
        temp_files.append(main_chart_path)
        render_overview_chart(df, mine_names, chart_type, poly_degree, series_details, main_chart_path)
        pdf.image(main_chart_path, x=10, w=190)
        pdf.ln(2)

        all_series = list(mine_names) + ["Total Output"]
        for i, s_name in enumerate(all_series):
            if pdf.get_y() > 210:
                pdf.add_page()
            fd_s, s_path = tempfile.mkstemp(suffix=".png")
            os.close(fd_s)
            temp_files.append(s_path)
            c = PALETTE[i % len(PALETTE)]
            render_single_series_chart(df, s_name, poly_degree, series_details[s_name], c, s_path)
            pdf.image(s_path, x=12, w=186)
            pdf.ln(2)

        pdf.add_page()
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 8, f"3. Individual Anomaly Sections ({len(anomalies)} Detected Events)", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(0, 6, "Each detected spike or drop is documented below with baseline comparison and test diagnostics.", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        if not anomalies:
            pdf.set_font("Helvetica", "I", 10)
            pdf.cell(0, 10, "No anomalies detected under the current test threshold settings.", new_x="LMARGIN", new_y="NEXT")
        else:
            for idx, anom in enumerate(anomalies, start=1):
                if pdf.get_y() > 238:
                    pdf.add_page()
                is_spike = anom["type"] == "Spike"
                if is_spike:
                    pdf.set_fill_color(254, 226, 226)
                    pdf.set_draw_color(220, 38, 38)
                    badge_text = "SPIKE ANOMALY"
                else:
                    pdf.set_fill_color(237, 233, 254)
                    pdf.set_draw_color(124, 58, 237)
                    badge_text = "DROP ANOMALY"

                pdf.set_font("Helvetica", "B", 10)
                pdf.set_text_color(15, 23, 42)
                header_line = f"Anomaly #{idx:02d}  |  [{badge_text}]  |  Date: {anom['date']}  |  Target: {anom['series']}"
                pdf.cell(0, 7, safe_text(header_line), border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

                pdf.set_fill_color(248, 250, 252)
                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(30, 41, 59)
                sign = "+" if anom["pct_from_ma"] >= 0 else ""
                metrics_line = (
                    f"Recorded Output: {anom['value']:.2f} tons   |   "
                    f"Moving Avg Baseline: {anom['moving_avg']:.2f} tons ({sign}{anom['pct_from_ma']:.1f}%)   |   "
                    f"Series Mean: {anom['mean']:.2f} tons   |   Z-Score: {anom['z_score']:+.2f}"
                )
                pdf.cell(0, 6.5, safe_text(metrics_line), border="LR", fill=True, new_x="LMARGIN", new_y="NEXT")

                pdf.set_font("Helvetica", "B", 8.8)
                tests_line = f"Triggered Detection Tests ({anom['test_count']}/4): {anom['tests_str']}"
                pdf.cell(0, 6.2, safe_text(tests_line), border="LR", fill=True, new_x="LMARGIN", new_y="NEXT")

                pdf.set_font("Helvetica", "", 8.2)
                pdf.set_text_color(71, 85, 105)
                if is_spike:
                    desc = (
                        f"Summary: Output on {anom['date']} ({anom['series']}) surged to {anom['value']:.2f}t, "
                        f"exceeding the moving average ({anom['moving_avg']:.2f}t) by {abs(anom['pct_from_ma']):.1f}%."
                    )
                else:
                    desc = (
                        f"Summary: Output on {anom['date']} ({anom['series']}) dropped to {anom['value']:.2f}t, "
                        f"falling {abs(anom['pct_from_ma']):.1f}% below the moving average ({anom['moving_avg']:.2f}t)."
                    )
                pdf.cell(0, 6.5, safe_text(desc), border="LRB", fill=True, new_x="LMARGIN", new_y="NEXT")
                pdf.ln(3)

        return bytes(pdf.output())
    finally:
        for p in temp_files:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
