# ======================================================================
# Financial Time-Series Anomaly Detection
# Complete, self-contained Python script
# Works with the “Yahoo Finance Stock Dataset” you described
# ======================================================================

# ----------------------------------------------------------------------
# 1. Imports
# ----------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from prophet import Prophet                     # pip install prophet
import ta                                       # pip install ta
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# 2. Configuration
# ----------------------------------------------------------------------
CSV_PATH   = "e:\AIML-INTERNSHIP-TASKS\Task1\stock_data.csv"
TICKERS    = ["AAPL", "AMZN", "TSLA", "GOOG"]  # subset; use None for all
CONTAM     = 0.01      # proportion of anomalies expected for Isolation Forest
FORECAST_HORIZON = 30  # days to forecast with Prophet

# ----------------------------------------------------------------------
# 3. Load & preprocess
# ----------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

# Keep selected tickers (optional)
if TICKERS:
    df = df[df["Company"].isin(TICKERS)]

# Parse dates & sort
df["Date"] = pd.to_datetime(df["Date"])
df.sort_values(["Company", "Date"], inplace=True)

# Handle missing values → forward-fill within each company
df = df.groupby("Company").apply(lambda x: x.ffill().bfill()).reset_index(drop=True)

# ----------------------------------------------------------------------
# 4. Technical indicators
# ----------------------------------------------------------------------
def add_indicators(group):
    """Add SMA, EMA, RSI, Bollinger Bands to one company’s DataFrame."""
    group = group.copy()
    close = group["Close"]

    group["SMA_20"] = ta.trend.sma_indicator(close, 20)
    group["EMA_20"] = ta.trend.ema_indicator(close, 20)
    group["RSI_14"] = ta.momentum.rsi(close, 14)

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    group["BB_high"] = bb.bollinger_hband()
    group["BB_low"]  = bb.bollinger_lband()
    return group

df = df.groupby("Company").apply(add_indicators).reset_index(drop=True)

# ----------------------------------------------------------------------
# 5. Unsupervised anomaly detection (Isolation Forest)
# ----------------------------------------------------------------------
def detect_anomaly(group):
    feats = group[["Close", "SMA_20", "EMA_20", "RSI_14"]].dropna()
    model = IsolationForest(n_estimators=200,
                            contamination=CONTAM,
                            random_state=42)
    preds = model.fit_predict(feats)          # -1 = anomaly, 1 = normal
    group["IF_anomaly"] = 0
    group.loc[feats.index, "IF_anomaly"] = preds
    return group

df = df.groupby("Company").apply(detect_anomaly).reset_index(drop=True)

# ----------------------------------------------------------------------
# 6. Time-series forecasting with Prophet & residual anomaly check
# ----------------------------------------------------------------------
def prophet_deviation(group):
    """Fit Prophet on closing prices and flag large residuals."""
    prop_df = group[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    model = Prophet(daily_seasonality=False, weekly_seasonality=True,
                    yearly_seasonality=True, interval_width=0.95)
    model.fit(prop_df)

    future = model.make_future_dataframe(periods=FORECAST_HORIZON)
    forecast = model.predict(future)

    # Merge forecast back to original dates only
    merged = group.merge(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
                         left_on="Date", right_on="ds", how="left")

    # Residuals & 3-σ rule
    residual = merged["Close"] - merged["yhat"]
    thresh   = 3 * residual.std()
    merged["Prophet_anomaly"] = np.where(np.abs(residual) > thresh, -1, 1)

    return merged.drop(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])

results = df.groupby("Company").apply(prophet_deviation).reset_index(drop=True)

# ----------------------------------------------------------------------
# 7. Visualization helper
# ----------------------------------------------------------------------
def plot_company(group, title_suffix=""):
    plt.figure(figsize=(14,6))
    plt.plot(group["Date"], group["Close"], label="Close")
    # Isolation Forest anomalies
    iso_pts = group[group["IF_anomaly"] == -1]
    plt.scatter(iso_pts["Date"], iso_pts["Close"], marker='o',
                edgecolor='red', facecolor='none', s=80,
                label="Isolation Forest")
    # Prophet residual anomalies
    prop_pts = group[group["Prophet_anomaly"] == -1]
    plt.scatter(prop_pts["Date"], prop_pts["Close"], marker='x',
                c='black', s=60, label="Prophet Residual")
    plt.title(f"{group['Company'].iloc[0]} Stock Price with Detected Anomalies {title_suffix}")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------
# 8. Generate plots & optional report
# ----------------------------------------------------------------------
for comp, grp in results.groupby("Company"):
    plot_company(grp)

# Save full results to CSV
results.to_csv("anomaly_detection_results.csv", index=False)
print("✅ Analysis complete. Detailed results saved to anomaly_detection_results.csv")
