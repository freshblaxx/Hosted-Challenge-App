import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import statsmodels.api as sm



# Page setup
st.set_page_config(
    layout="wide",
    page_title="Whirlpool — Sales and Price Optimization Dashboard"
)


# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv("/Users/domo/Library/Mobile Documents/com~apple~CloudDocs/Desktop/University/Master/Semester 3/Courses/Data Analytics/code/II/project/final shit/ML_Clustered_Database_Horizon_Global_Consulting.csv")
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
    st.markdown("""
    ### Overview  
    Explore Whirlpool sales performance with comparisons and time-series views.
    """)

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

    # KPI summary metrics
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Gross Sales", f"${df_filtered['gross_sales'].sum():,.0f}")
    kpi2.metric("Average Final Price", f"${df_filtered['price_final'].mean():.2f}")
    kpi3.metric("Total Demand", f"{df_filtered['demand'].sum():,.0f}")
    kpi4.metric("Average DCM", f"{df_filtered['dcm'].mean():.2f}")

    st.subheader("Sales Performance Comparison")

    view_choice = st.selectbox(
        "View Sales By",
        ["Product Type", "Trade Partner", "SKU"]
    )

    if view_choice == "Product Type":
        df_view = (
            df_filtered.groupby("product_type", as_index=False)["gross_sales"]
            .sum()
            .sort_values("gross_sales", ascending=False)
            .rename(columns={"product_type": "Category"})
        )
        y_col = "Category"

    elif view_choice == "Trade Partner":
        df_view = (
            df_filtered.groupby("trade_partner", as_index=False)["gross_sales"]
            .sum()
            .sort_values("gross_sales", ascending=False)
            .rename(columns={"trade_partner": "Trade Partner"})
        )
        y_col = "Trade Partner"

    else:
        df_view = (
            df_filtered.groupby("sku", as_index=False)["gross_sales"]
            .sum()
            .sort_values("gross_sales", ascending=False)
            .head(20)
            .rename(columns={"sku": "SKU"})
        )
        y_col = "SKU"

    chart_combined = (
        alt.Chart(df_view)
        .mark_bar()
        .encode(
            x=alt.X("gross_sales:Q", title="Total Gross Sales"),
            y=alt.Y(f"{y_col}:N", sort="-x"),
            tooltip=[y_col, "gross_sales"]
        )
        .properties(height=380)
    )

    st.altair_chart(chart_combined, use_container_width=True)

    st.subheader("Gross Sales Over Time")

    df_time = (
        df_filtered.groupby("date", as_index=False)["gross_sales"]
        .sum()
        .sort_values("date")
    )

    chart_time = (
        alt.Chart(df_time)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("gross_sales:Q", title="Gross Sales"),
            tooltip=["date", "gross_sales"]
        )
        .properties(height=350)
    )

    st.altair_chart(chart_time, use_container_width=True)


with tab2:
    st.header("Price Optimization — Maximize DCM for a SKU")

    st.markdown("""
    This section uses an XGBoost model to estimate how pricing affects DCM  
    and suggests a price that may maximize performance for the selected SKU.
    """)

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.8,
    )

    model.fit(X_train, y_train)

    price_range = np.linspace(X["price_final"].min(), X["price_final"].max(), 80)
    dcm_pred = model.predict(price_range.reshape(-1, 1))

    optimal_price = price_range[np.argmax(dcm_pred)]
    optimal_dcm = dcm_pred.max()

    m1, m2 = st.columns(2)
    m1.metric("Optimal Price", f"${optimal_price:.2f}")
    m2.metric("Max Predicted DCM", f"{optimal_dcm:.2f}")

    df_pred_curve = pd.DataFrame({
        "price_final": price_range,
        "predicted_dcm": dcm_pred
    })

    chart_price_curve = (
        alt.Chart(df_pred_curve)
        .mark_line()
        .encode(
            x=alt.X("price_final:Q", title="Final Price"),
            y=alt.Y("predicted_dcm:Q", title="Predicted DCM"),
            tooltip=["price_final", "predicted_dcm"]
        )
        .properties(height=320)
    )

    st.altair_chart(chart_price_curve, use_container_width=True)

    st.subheader("DCM Forecast Over Time (Prophet)")

    df_prophet = (
        df_model[["date", "dcm"]]
        .rename(columns={"date": "ds", "dcm": "y"})
        .dropna()
    )

    if len(df_prophet) >= 10:
        m = Prophet(yearly_seasonality=True)
        m.fit(df_prophet)

        future = m.make_future_dataframe(periods=60)
        forecast = m.predict(future)

        plot_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

        line = (
            alt.Chart(plot_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("ds:T", title="Date"),
                y=alt.Y("yhat:Q", title="Predicted DCM"),
                tooltip=[
                    alt.Tooltip("ds:T", title="Date"),
                    alt.Tooltip("yhat:Q", title="Predicted DCM"),
                    alt.Tooltip("yhat_lower:Q", title="Lower CI"),
                    alt.Tooltip("yhat_upper:Q", title="Upper CI"),
                ]
            )
            .properties(height=260)
        )

        band = (
            alt.Chart(plot_df)
            .mark_area(opacity=0.25)
            .encode(
                x="ds:T",
                y="yhat_lower:Q",
                y2="yhat_upper:Q"
            )
        )

        st.altair_chart(band + line, use_container_width=True)

    else:
        st.info("Not enough time-series points to train Prophet.")
