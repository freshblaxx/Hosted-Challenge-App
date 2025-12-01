import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings("ignore")

# Page setup
st.set_page_config(
    layout="wide",
    page_title="Whirlpool — Sales and Price Optimization Dashboard"
)

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv("ML_Clustered_Database_Horizon_Global_Consulting.csv")
    df.columns = [c.strip() for c in df.columns]

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_cols = [
        "inventory", "quantity", "gross_sales", "year", "month", "quarter",
        "iso_week", "cpi", "price_list", "price_final", "vpc", "wty",
        "varfw", "varsga", "usd_to_mxn", "total_variable_cost",
        "real_discount_pct", "dcm", "tp_pt", "tp_sku", "demand"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["gross_sales"] = df["gross_sales"].fillna(0)
    df["price_final"] = df["price_final"].fillna(df["price_final"].median())
    df["dcm"] = df["dcm"].fillna(0)
    df["demand"] = df["demand"].fillna(0)

    return df

df = load_data()

# Main layout
st.title("Whirlpool — Sales Analytics and Price Optimization")

tab1, tab2 = st.tabs(["Sales Dashboard", "Price Optimization"])

with tab1:
    st.markdown("### Overview  \nExplore Whirlpool sales performance with comparisons and time-series views.")

    st.subheader("Date Range Filter")
    if "date" in df.columns:
        min_date = df["date"].min()
        max_date = df["date"].max()

        date_range = st.date_input(
            "Select Date Range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date()
        )
    else:
        st.error("The dataset must contain a 'date' column.")
        st.stop()

    df_filtered = df[
        (df["date"].dt.date >= date_range[0]) &
        (df["date"].dt.date <= date_range[1])
    ].copy()

    # KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Gross Sales", f"${df_filtered['gross_sales'].sum():,.0f}")
    kpi2.metric("Average Final Price", f"${df_filtered['price_final'].mean():.2f}")
    kpi3.metric("Total Demand", f"{df_filtered['demand'].sum():,.0f}")
    kpi4.metric("Average DCM", f"{df_filtered['dcm'].mean():.2f}")

    st.subheader("Sales Performance Comparison")
    view_choice = st.selectbox("View Sales By", ["Product Type", "Trade Partner", "SKU"])

    if view_choice == "Product Type":
        df_view = df_filtered.groupby("product_type", as_index=False)["gross_sales"].sum().sort_values("gross_sales", ascending=False).rename(columns={"product_type": "Category"})
        y_col = "Category"
    elif view_choice == "Trade Partner":
        df_view = df_filtered.groupby("trade_partner", as_index=False)["gross_sales"].sum().sort_values("gross_sales", ascending=False).rename(columns={"trade_partner": "Trade Partner"})
        y_col = "Trade Partner"
    else:
        df_view = df_filtered.groupby("sku", as_index=False)["gross_sales"].sum().sort_values("gross_sales", ascending=False).head(20).rename(columns={"sku": "SKU"})
        y_col = "SKU"

    chart_combined = alt.Chart(df_view).mark_bar().encode(
        x=alt.X("gross_sales:Q", title="Total Gross Sales"),
        y=alt.Y(f"{y_col}:N", sort="-x"),
        tooltip=[y_col, "gross_sales"]
    ).properties(height=380)

    st.altair_chart(chart_combined, use_container_width=True)

    st.subheader("Gross Sales Over Time")
    df_time = df_filtered.groupby("date", as_index=False)["gross_sales"].sum().sort_values("date")

    chart_time = alt.Chart(df_time).mark_line(point=True).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("gross_sales:Q", title="Gross Sales"),
        tooltip=["date", "gross_sales"]
    ).properties(height=350)

    st.altair_chart(chart_time, use_container_width=True)


with tab2:
    st.header("Price Optimization — Maximize DCM for a SKU")
    st.markdown("This section uses XGBoost to estimate optimal pricing and forecasts future DCM using Exponential Smoothing (robust & fast).")

    trade_counts = df["trade_partner"].value_counts()
    all_trade_partners = trade_counts.index.tolist()
    tp_for_model = st.selectbox("Trade Partner", all_trade_partners)

    sku_counts = df[df["trade_partner"] == tp_for_model]["sku"].value_counts()
    all_skus = sku_counts.index.tolist()
    sku_for_model = st.selectbox("SKU", all_skus)

    df_model = df[
        (df["trade_partner"] == tp_for_model) &
        (df["sku"] == sku_for_model)
    ].dropna(subset=["price_final", "dcm"])

    if len(df_model) < 20:
        st.warning("Not enough data for this SKU and trade partner. Try another choice.")
        st.stop()

    st.subheader("Predicted DCM vs Price (XGBoost)")
    X = df_model[["price_final"]]
    y = df_model["dcm"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.9, colsample_bytree=0.8, random_state=42)
    model.fit(X_train, y_train)

    price_range = np.linspace(X["price_final"].min(), X["price_final"].max(), 80)
    dcm_pred = model.predict(price_range.reshape(-1, 1))

    optimal_price = price_range[np.argmax(dcm_pred)]
    optimal_dcm = dcm_pred.max()

    m1, m2 = st.columns(2)
    m1.metric("Optimal Price", f"${optimal_price:,.2f}")
    m2.metric("Max Predicted DCM", f"${optimal_dcm:,.2f}")

    df_pred_curve = pd.DataFrame({"price_final": price_range, "predicted_dcm": dcm_pred})

    chart_price_curve = alt.Chart(df_pred_curve).mark_line().encode(
        x=alt.X("price_final:Q", title="Final Price"),
        y=alt.Y("predicted_dcm:Q", title="Predicted DCM"),
        tooltip=["price_final", "predicted_dcm"]
    ).properties(height=320)

    st.altair_chart(chart_price_curve, use_container_width=True)

    st.subheader("DCM Forecast Over Time (Exponential Smoothing)")

    ts_data = df_model[["date", "dcm"]].copy()
    ts_data = ts_data.rename(columns={"date": "ds", "dcm": "y"})
    ts_data = ts_data.dropna().sort_values("ds")
    ts_data = ts_data.set_index("ds")["y"]

    if len(ts_data) >= 12:
        # Resample to monthly for smoother, more reliable forecast
        ts_monthly = ts_data.resample("M").mean().fillna(method="ffill")

        try:
            model_ets = ExponentialSmoothing(
                ts_monthly,
                trend="add",
                seasonal="add",
                seasonal_periods=12
            )
            fit = model_ets.fit()

            forecast_steps = 60
            forecast = fit.forecast(forecast_steps)
            forecast_index = pd.date_range(start=ts_monthly.index[-1] + pd.offsets.MonthBegin(1), periods=forecast_steps, freq="M")

            # Combine historical + forecast
            plot_df = pd.DataFrame({
                "ds": list(ts_monthly.index) + list(forecast_index),
                "value": list(ts_monthly.values) + list(forecast),
                "type": ["Historical"] * len(ts_monthly) + ["Forecast"] * forecast_steps
            })

            # Approximate confidence interval
            se = np.std(fit.resid)
            plot_df["lower"] = plot_df["value"] - 1.96 * se
            plot_df["upper"] = plot_df["value"] + 1.96 * se

            line = alt.Chart(plot_df).mark_line().encode(
                x="ds:T",
                y=alt.Y("value:Q", title="DCM"),
                color=alt.Color("type:N", scale=alt.Scale(domain=["Historical", "Forecast"], range=["#1f77b4", "#ff7f0e"])),
                tooltip=["ds", "value", "type"]
            )

            band = alt.Chart(plot_df[plot_df["type"] == "Forecast"]).mark_area(opacity=0.2).encode(
                x="ds:T",
                y="lower:Q",
                y2="upper:Q"
            )

            st.altair_chart(band + line, use_container_width=True)

        except Exception as e:
            st.error(f"Forecasting failed: {str(e)}. Try a different SKU.")
    else:
        st.info("Not enough data points (need at least 12 months) for reliable forecasting.")