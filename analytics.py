import numpy as np
import pandas as pd
from scipy import stats


def compute_summary_stats(df, series_names):
    rows = []
    for name in series_names:
        vals = df[name].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        q75, q25 = np.percentile(vals, [75, 25])
        rows.append({
            "Mine / Series": name,
            "Mean": round(float(np.mean(vals)), 2),
            "Std Dev": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 2),
            "Median": round(float(np.median(vals)), 2),
            "IQR": round(float(q75 - q25), 2),
            "Min": round(float(np.min(vals)), 2),
            "Max": round(float(np.max(vals)), 2),
        })
    return pd.DataFrame(rows)


def fit_polynomial_trend(values, degree):
    y = np.asarray(values, dtype=float)
    n = len(y)
    if n == 0:
        return np.array([])
    x = np.arange(n, dtype=float)
    deg = max(1, min(int(degree), 4, n - 1))
    coeffs = np.polyfit(x, y, deg)
    return np.polyval(coeffs, x)


def grubbs_outliers_mask(values, alpha=0.05):
    y = np.asarray(values, dtype=float)
    n_total = len(y)
    mask = np.zeros(n_total, dtype=bool)
    if n_total < 7:
        return mask

    remaining_idx = list(range(n_total))
    while len(remaining_idx) >= 7:
        sub = y[remaining_idx]
        n = len(sub)
        mean = np.mean(sub)
        std = np.std(sub, ddof=1)
        if std <= 1e-9:
            break
        abs_dev = np.abs(sub - mean)
        max_pos = int(np.argmax(abs_dev))
        g_stat = abs_dev[max_pos] / std

        t_dist = stats.t.ppf(1.0 - alpha / (2.0 * n), n - 2)
        g_crit = ((n - 1) / np.sqrt(n)) * np.sqrt((t_dist ** 2) / (n - 2 + t_dist ** 2))

        if g_stat > g_crit:
            orig_i = remaining_idx[max_pos]
            mask[orig_i] = True
            remaining_idx.pop(max_pos)
        else:
            break

    return mask


def compute_moving_baseline(values, window):
    s = pd.Series(values, dtype=float)
    w = max(3, int(window))
    rolling_med = s.rolling(window=w, center=True, min_periods=1).median()
    rolling_mean = s.rolling(window=w, center=True, min_periods=1).mean()
    baseline = 0.6 * rolling_med + 0.4 * rolling_mean
    return baseline.to_numpy(dtype=float)


def detect_series_anomalies(df, series_name, params):
    dates = df["Date"].dt.strftime("%Y-%m-%d").tolist()
    vals = df[series_name].to_numpy(dtype=float)
    n = len(vals)
    if n == 0:
        return [], {}

    q75, q25 = np.percentile(vals, [75, 25])
    iqr = q75 - q25
    iqr_k = float(params.get("iqr_k", 1.5))
    iqr_low = q25 - iqr_k * iqr
    iqr_high = q75 + iqr_k * iqr
    iqr_mask = (vals < iqr_low) | (vals > iqr_high)

    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    z_thresh = float(params.get("z_threshold", 2.0))
    if std_val > 1e-9:
        z_scores = (vals - mean_val) / std_val
    else:
        z_scores = np.zeros(n, dtype=float)
    z_mask = np.abs(z_scores) > z_thresh

    ma_window = int(params.get("ma_window", 7))
    ma_pct = float(params.get("ma_pct", 25.0))
    ma_vals = compute_moving_baseline(vals, ma_window)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_from_ma = np.where(np.abs(ma_vals) > 1e-9, ((vals - ma_vals) / ma_vals) * 100.0, 0.0)
    ma_mask = np.abs(pct_from_ma) > ma_pct

    grubbs_alpha = float(params.get("grubbs_alpha", 0.05))
    grubbs_mask = grubbs_outliers_mask(vals, alpha=grubbs_alpha)

    anomalies = []
    for i in range(n):
        triggered = []
        if iqr_mask[i]:
            triggered.append("IQR Rule")
        if z_mask[i]:
            triggered.append("Z-Score")
        if ma_mask[i]:
            triggered.append("Moving Avg")
        if grubbs_mask[i]:
            triggered.append("Grubbs' Test")

        if triggered:
            ref = ma_vals[i] if abs(vals[i] - ma_vals[i]) >= abs(vals[i] - mean_val) * 0.5 else mean_val
            kind = "Spike" if vals[i] >= ref else "Drop"
            anomalies.append({
                "index": i,
                "date": dates[i],
                "series": series_name,
                "value": round(float(vals[i]), 2),
                "mean": round(mean_val, 2),
                "moving_avg": round(float(ma_vals[i]), 2),
                "pct_from_ma": round(float(pct_from_ma[i]), 1),
                "z_score": round(float(z_scores[i]), 2),
                "type": kind,
                "tests": triggered,
                "tests_str": ", ".join(triggered),
                "test_count": len(triggered),
            })

    details = {
        "iqr_mask": iqr_mask,
        "z_mask": z_mask,
        "ma_mask": ma_mask,
        "grubbs_mask": grubbs_mask,
        "moving_avg": ma_vals,
        "z_scores": z_scores,
        "pct_from_ma": pct_from_ma,
    }
    return anomalies, details


def filter_anomalies_by_mode(anomalies, mode):
    if mode == "Any Test (Union)":
        return anomalies
    if mode == "Consensus (2+ Tests)":
        return [a for a in anomalies if a["test_count"] >= 2]
    mapping = {
        "IQR Rule": "IQR Rule",
        "Z-Score": "Z-Score",
        "Moving Average (%)": "Moving Avg",
        "Grubbs' Test": "Grubbs' Test",
    }
    target = mapping.get(mode)
    if not target:
        return anomalies
    return [a for a in anomalies if target in a["tests"]]


def analyze_all(df, mine_names, params, highlight_mode="Any Test (Union)"):
    all_series = list(mine_names) + ["Total Output"]
    stats_df = compute_summary_stats(df, all_series)
    all_anomalies = []
    series_details = {}

    for s in all_series:
        anoms, det = detect_series_anomalies(df, s, params)
        filtered = filter_anomalies_by_mode(anoms, highlight_mode)
        all_anomalies.extend(filtered)
        series_details[s] = {
            "all_anomalies": anoms,
            "filtered_anomalies": filtered,
            "details": det,
        }

    all_anomalies.sort(key=lambda x: (x["date"], x["series"]))
    return stats_df, all_anomalies, series_details
