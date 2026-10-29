import csv
import io
import os
import re
import pandas as pd
import requests

DEFAULT_SHEET_ID = "11G48bu_26GeUaM1KEEGVjg5l27gLpx1J9kvh1CuUxzA"
DEFAULT_SHEET_URL = f"https://docs.google.com/spreadsheets/d/{DEFAULT_SHEET_ID}/edit"
SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), "data_snapshot.csv")


def extract_sheet_id(url_or_id):
    text = (url_or_id or "").strip()
    if not text:
        return DEFAULT_SHEET_ID
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", text)
    if match:
        return match.group(1)
    return text


def fetch_csv_text(url_or_id):
    sheet_id = extract_sheet_id(url_or_id)
    public_urls = [
        f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid=0",
        f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet=Simulator",
    ]
    for export_url in public_urls:
        try:
            resp = requests.get(export_url, timeout=8)
            if resp.status_code == 200 and "Date" in resp.text and "<!DOCTYPE html>" not in resp.text:
                return resp.text, "Live Google Sheet"
        except Exception:
            pass

    if os.path.exists(SNAPSHOT_PATH):
        with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
            return f.read(), "Google Sheet Snapshot"

    raise RuntimeError("Could not load data from Google Sheets or local snapshot.")


def parse_simulator_csv(csv_text):
    rows = list(csv.reader(io.StringIO(csv_text)))
    max_cols = max((len(r) for r in rows), default=8)
    padded_rows = [r + [""] * (max_cols - len(r)) for r in rows]
    raw_df = pd.DataFrame(padded_rows).fillna("")
    meta = {
        "start_date": "",
        "days_count": "",
        "distribution": "",
        "param1_name": "Mean",
        "param1_val": "",
        "param2_name": "Deviation",
        "param2_val": "",
        "correlation": "",
        "daily_growth": "",
        "weekday_factors": [],
        "events": [],
    }

    header_row_idx = None
    for idx, row in raw_df.iterrows():
        first_cell = str(row.iloc[0]).strip()
        if first_cell == "Range:" and idx + 1 < len(raw_df):
            next_row = raw_df.iloc[idx + 1]
            meta["start_date"] = str(next_row.iloc[1]).strip()
            meta["days_count"] = str(next_row.iloc[2]).strip()
        elif first_cell == "Random:" and idx + 1 < len(raw_df):
            meta["param1_name"] = str(row.iloc[2]).strip() or "Param 1"
            meta["param2_name"] = str(row.iloc[3]).strip() or "Param 2"
            next_row = raw_df.iloc[idx + 1]
            meta["distribution"] = str(next_row.iloc[1]).strip()
            meta["param1_val"] = str(next_row.iloc[2]).strip()
            meta["param2_val"] = str(next_row.iloc[3]).strip()
            meta["correlation"] = str(next_row.iloc[4]).strip()
        elif first_cell == "Daily output:":
            meta["weekday_factors"] = [str(x).strip() for x in row.iloc[1:8].tolist() if str(x).strip() != ""]
        elif first_cell == "Daily growth:":
            meta["daily_growth"] = str(row.iloc[1]).strip()
        elif first_cell == "Events:":
            for ev_i in range(idx + 1, min(idx + 7, len(raw_df))):
                ev_row = raw_df.iloc[ev_i]
                if str(ev_row.iloc[0]).strip() != "":
                    break
                ev_day = str(ev_row.iloc[1]).strip()
                if ev_day:
                    meta["events"].append({
                        "day": ev_day,
                        "duration": str(ev_row.iloc[2]).strip(),
                        "factor": str(ev_row.iloc[3]).strip(),
                        "probability": str(ev_row.iloc[4]).strip(),
                        "affected_mines": str(ev_row.iloc[5]).strip() if len(ev_row) > 5 else "",
                    })
        elif first_cell == "Date":
            header_row_idx = idx
            break

    if header_row_idx is None:
        raise ValueError("Could not find Date header row in spreadsheet.")

    headers = [str(col).strip() for col in raw_df.iloc[header_row_idx].tolist()]
    data_rows = raw_df.iloc[header_row_idx + 1 :].copy()

    if len(headers) >= 3 and headers[0] == "Date" and headers[1] == "Mine":
        long_df = pd.DataFrame({
            "Date": data_rows.iloc[:, 0].astype(str).str.strip(),
            "Mine": data_rows.iloc[:, 1].astype(str).str.strip(),
            "Output": pd.to_numeric(data_rows.iloc[:, 2], errors="coerce"),
        })
        long_df = long_df[(long_df["Date"] != "") & (long_df["Mine"] != "")].copy()
        long_df["Date"] = pd.to_datetime(long_df["Date"], errors="coerce")
        long_df = long_df.dropna(subset=["Date", "Output"]).reset_index(drop=True)

        mine_order = []
        for m in long_df["Mine"].tolist():
            if m not in mine_order:
                mine_order.append(m)

        pivoted = long_df.pivot_table(index="Date", columns="Mine", values="Output", aggfunc="sum").reset_index()
        pivoted = pivoted.sort_values("Date").reset_index(drop=True)
        active_mines = [m for m in mine_order if m in pivoted.columns]
        df = pivoted[["Date"] + active_mines].fillna(0.0).copy()
        df["Total Output"] = df[active_mines].sum(axis=1).round(2)
        meta["records_df"] = long_df.assign(Date=long_df["Date"].dt.strftime("%Y-%m-%d"))
        return df, active_mines, meta

    data_rows.columns = headers
    valid_cols = ["Date"]
    mine_names = []
    for c in headers[1:]:
        if c and c not in ("Total", "Date"):
            valid_cols.append(c)
            mine_names.append(c)

    df = data_rows[valid_cols].copy()
    df["Date"] = df["Date"].astype(str).str.strip()
    df = df[df["Date"] != ""].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    active_mines = []
    for m in mine_names:
        df[m] = pd.to_numeric(df[m], errors="coerce")
        if df[m].notna().sum() > 0:
            df[m] = df[m].fillna(0.0)
            active_mines.append(m)

    df = df[["Date"] + active_mines].copy()
    df["Total Output"] = df[active_mines].sum(axis=1).round(2)
    return df, active_mines, meta
