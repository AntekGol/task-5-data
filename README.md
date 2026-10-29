Dashboard for Weyland-Yutani mine extraction data pulled from Google Sheets, made with streamlit, pandas and plotly. Calculates basic stats, anomaly detection tests (IQR, z-score, moving average, Grubbs) with polynomial trendlines and exports a PDF report.

To run locally:

pip install -r requirements.txt
streamlit run app.py
