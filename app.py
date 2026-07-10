"""
End-to-End Sales Forecasting & Demand Intelligence System — Streamlit Dashboard
Run locally with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Sales Forecasting & Demand Intelligence", layout="wide")


# ------------------------------------------------------------------
# Data loading & caching
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv", encoding="utf-8-sig")
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    return df


@st.cache_data
def get_monthly_series(df, category=None, region=None):
    d = df.copy()
    if category and category != "All":
        d = d[d["Category"] == category]
    if region and region != "All":
        d = d[d["Region"] == region]
    ts = d.set_index("Order Date").resample("MS")["Sales"].sum()
    ts.index.freq = "MS"
    return ts


@st.cache_data
def get_weekly_series(df):
    ts = df.set_index("Order Date").resample("W")["Sales"].sum()
    return ts


def sarima_forecast(ts, horizon):
    ts = ts.asfreq("MS").fillna(0)
    try:
        model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                         enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False)
    except Exception:
        model = SARIMAX(ts, order=(1, 1, 0), enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False)
    fc = fit.get_forecast(steps=horizon)
    return fc.predicted_mean, fc.conf_int()


def eval_metrics(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    return mae, rmse


df = load_data()

st.sidebar.title("📊 Sales Intelligence")
page = st.sidebar.radio("Navigate", [
    "Sales Overview", "Forecast Explorer", "Anomaly Report", "Demand Segments"
])

# ------------------------------------------------------------------
# Page 1 — Sales Overview Dashboard
# ------------------------------------------------------------------
if page == "Sales Overview":
    st.title("Sales Overview Dashboard")

    col1, col2 = st.columns(2)
    with col1:
        regions = ["All"] + sorted(df["Region"].unique().tolist())
        sel_region = st.selectbox("Filter by Region", regions)
    with col2:
        categories = ["All"] + sorted(df["Category"].unique().tolist())
        sel_category = st.selectbox("Filter by Category", categories)

    filtered = df.copy()
    if sel_region != "All":
        filtered = filtered[filtered["Region"] == sel_region]
    if sel_category != "All":
        filtered = filtered[filtered["Category"] == sel_category]

    st.metric("Total Sales (filtered)", f"${filtered['Sales'].sum():,.0f}")

    st.subheader("Total Sales by Year")
    yearly = filtered.groupby("Year")["Sales"].sum()
    st.bar_chart(yearly)

    st.subheader("Monthly Sales Trend")
    monthly = filtered.set_index("Order Date").resample("MS")["Sales"].sum()
    st.line_chart(monthly)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Sales by Region")
        st.bar_chart(filtered.groupby("Region")["Sales"].sum())
    with col4:
        st.subheader("Sales by Category")
        st.bar_chart(filtered.groupby("Category")["Sales"].sum())


# ------------------------------------------------------------------
# Page 2 — Forecast Explorer
# ------------------------------------------------------------------
elif page == "Forecast Explorer":
    st.title("Forecast Explorer")

    dim = st.selectbox("Select dimension", ["Category", "Region"])
    options = sorted(df[dim].unique().tolist())
    selected = st.selectbox(f"Select {dim}", options)
    horizon = st.select_slider("Forecast horizon (months ahead)", options=[1, 2, 3], value=3)

    if dim == "Category":
        ts = get_monthly_series(df, category=selected)
    else:
        ts = get_monthly_series(df, region=selected)

    if len(ts) < 15:
        st.warning("Not enough monthly history for this segment to forecast reliably.")
    else:
        train_ts = ts.iloc[:-3] if len(ts) > 15 else ts
        test_ts = ts.iloc[-3:] if len(ts) > 15 else None

        with st.spinner("Fitting SARIMA model..."):
            forecast, ci = sarima_forecast(train_ts, 3)
            future_forecast, future_ci = sarima_forecast(ts, horizon)

        mae, rmse = eval_metrics(test_ts.values, forecast.values) if test_ts is not None else (None, None)

        fig, ax = plt.subplots(figsize=(10, 4))
        ts.plot(ax=ax, label="Actual", color="#4C72B0")
        future_forecast.plot(ax=ax, label=f"{horizon}-Month Forecast", color="#C44E52", marker="o")
        ax.fill_between(future_ci.index, future_ci.iloc[:, 0], future_ci.iloc[:, 1],
                         color="#C44E52", alpha=0.2)
        ax.set_title(f"SARIMA Forecast — {selected} ({dim})")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Forecast values")
        st.dataframe(future_forecast.rename("Forecasted Sales ($)").to_frame())

        if mae is not None:
            col1, col2 = st.columns(2)
            col1.metric("MAE (holdout backtest)", f"${mae:,.0f}")
            col2.metric("RMSE (holdout backtest)", f"${rmse:,.0f}")


# ------------------------------------------------------------------
# Page 3 — Anomaly Report
# ------------------------------------------------------------------
elif page == "Anomaly Report":
    st.title("Anomaly Report")

    weekly_ts = get_weekly_series(df)

    iso = IsolationForest(contamination=0.08, random_state=42)
    iso_pred = iso.fit_predict(weekly_ts.values.reshape(-1, 1))
    anomalies_iso = weekly_ts[iso_pred == -1]

    rolling_mean = weekly_ts.rolling(window=8, min_periods=1).mean()
    rolling_std = weekly_ts.rolling(window=8, min_periods=1).std().fillna(weekly_ts.std())
    z_scores = (weekly_ts - rolling_mean) / rolling_std
    anomalies_z = weekly_ts[z_scores.abs() > 2]

    method = st.radio("Detection method", ["Isolation Forest", "Z-Score (rolling)"], horizontal=True)
    anomalies = anomalies_iso if method == "Isolation Forest" else anomalies_z

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(weekly_ts.index, weekly_ts.values, color="#4C72B0", label="Weekly Sales")
    ax.scatter(anomalies.index, anomalies.values, color="red", s=60, zorder=5, label="Anomaly")
    ax.set_title(f"Anomaly Detection — {method}")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Detected anomaly weeks")
    st.dataframe(anomalies.sort_values(ascending=False).rename("Sales ($)").to_frame())


# ------------------------------------------------------------------
# Page 4 — Product Demand Segments
# ------------------------------------------------------------------
elif page == "Demand Segments":
    st.title("Product Demand Segments")

    subcat = df.groupby("Sub-Category").agg(
        Total_Sales=("Sales", "sum"),
        Avg_Order_Value=("Sales", "mean"),
    ).reset_index()

    subcat_year = df.groupby(["Sub-Category", "Year"])["Sales"].sum().unstack("Year")
    subcat["Growth_Rate_%"] = subcat["Sub-Category"].map(
        (subcat_year[subcat_year.columns[-1]] - subcat_year[subcat_year.columns[0]])
        / subcat_year[subcat_year.columns[0]] * 100
    )
    subcat_monthly = df.set_index("Order Date").groupby("Sub-Category").resample("MS")["Sales"].sum()
    subcat["Volatility"] = subcat["Sub-Category"].map(subcat_monthly.groupby("Sub-Category").std())
    subcat = subcat.dropna()

    features = ["Total_Sales", "Growth_Rate_%", "Volatility", "Avg_Order_Value"]
    X_scaled = StandardScaler().fit_transform(subcat[features].values)

    k = st.slider("Number of clusters (k)", 2, 6, 4)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    subcat["Cluster"] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    subcat["PCA1"], subcat["PCA2"] = X_pca[:, 0], X_pca[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    for c in sorted(subcat["Cluster"].unique()):
        subset = subcat[subcat["Cluster"] == c]
        ax.scatter(subset["PCA1"], subset["PCA2"], label=f"Cluster {c}", s=90)
        for _, r in subset.iterrows():
            ax.annotate(r["Sub-Category"], (r["PCA1"], r["PCA2"]), fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
    ax.set_title("Sub-Category Demand Clusters (PCA Projection)")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Sub-categories by cluster")
    st.dataframe(subcat[["Sub-Category", "Cluster", "Total_Sales", "Growth_Rate_%", "Volatility"]]
                 .sort_values("Cluster").reset_index(drop=True))
