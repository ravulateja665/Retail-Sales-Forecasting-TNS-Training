
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page configuration
st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📊",
    layout="wide"
)

# Load dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(
    os.path.join(BASE_DIR, "retail_store_inventory.csv")
)

df["Date"] = pd.to_datetime(
    df["Date"], format="%d-%m-%Y"
)

# Dashboard title
st.title("📊 Retail Sales Forecasting Dashboard")
st.write("Analyze retail sales and predict future product demand.")

st.divider()

# Key performance indicators
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Units Sold", f"{df['Units Sold'].sum():,}")
col2.metric("Total Records", f"{len(df):,}")
col3.metric("Total Stores", df["Store ID"].nunique())
col4.metric("Product Categories", df["Category"].nunique())

st.divider()

# Monthly sales
st.subheader("1. Monthly Sales Analysis")

monthly_sales = df.groupby(
    df["Date"].dt.to_period("M")
)["Units Sold"].sum()

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(monthly_sales.index.astype(str), monthly_sales.values)
ax.set_xlabel("Month")
ax.set_ylabel("Units Sold")
ax.set_title("Monthly Retail Sales")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Category-wise sales
st.subheader("2. Category-wise Sales")

category_sales = df.groupby("Category")["Units Sold"].sum()

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(category_sales.index, category_sales.values)
ax.set_xlabel("Category")
ax.set_ylabel("Units Sold")
ax.set_title("Sales by Product Category")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Region-wise sales
st.subheader("3. Region-wise Sales")

region_sales = df.groupby("Region")["Units Sold"].sum()

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(region_sales.index, region_sales.values)
ax.set_xlabel("Region")
ax.set_ylabel("Units Sold")
ax.set_title("Sales by Region")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Seasonal sales
st.subheader("4. Seasonal Sales Analysis")

seasonal_sales = df.groupby("Seasonality")["Units Sold"].sum()

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(seasonal_sales.index, seasonal_sales.values)
ax.set_xlabel("Season")
ax.set_ylabel("Units Sold")
ax.set_title("Sales by Season")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Actual sales vs demand forecast
st.subheader("5. Actual Sales vs Demand Forecast")

daily_actual = df.groupby("Date")["Units Sold"].sum()
daily_forecast = df.groupby("Date")["Demand Forecast"].sum()

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(daily_actual.index, daily_actual.values, label="Actual Sales")
ax.plot(daily_forecast.index, daily_forecast.values, label="Demand Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Units Sold")
ax.legend()
ax.set_title("Actual Sales vs Demand Forecast")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Machine learning model
st.subheader("6. Random Forest Model Evaluation")

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek

features = [
    "Year", "Month", "Day", "DayOfWeek",
    "Inventory Level", "Price", "Discount",
    "Competitor Pricing", "Holiday/Promotion"
]

X = pd.get_dummies(
    df[features],
    columns=["Holiday/Promotion"],
    dtype=int
)

y = df["Units Sold"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=150,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"{mae:.2f}")
m2.metric("RMSE", f"{rmse:.2f}")
m3.metric("R² Score", f"{r2:.3f}")

# Feature importance
st.subheader("7. Feature Importance")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(importance["Feature"], importance["Importance"])
ax.set_xlabel("Importance")
ax.set_title("Factors Influencing Retail Sales")
ax.invert_yaxis()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Future sales prediction
st.subheader("8. Future Sales Prediction")

st.write("Enter the details to predict expected units sold.")

c1, c2, c3 = st.columns(3)

with c1:
    future_date = st.date_input("Prediction Date")

with c2:
    inventory = st.number_input(
        "Inventory Level", min_value=0, max_value=10000, value=500
    )

with c3:
    price = st.number_input(
        "Product Price", min_value=0.0, value=50.0
    )

c4, c5 = st.columns(2)

with c4:
    discount = st.number_input(
        "Discount", min_value=0, max_value=100, value=0
    )

with c5:
    competitor_price = st.number_input(
        "Competitor Pricing", min_value=0.0, value=50.0
    )

holiday = st.selectbox(
    "Holiday / Promotion",
    options=sorted(df["Holiday/Promotion"].unique())
)

if st.button("Predict Future Sales"):
    future_input = pd.DataFrame({
        "Year": [future_date.year],
        "Month": [future_date.month],
        "Day": [future_date.day],
        "DayOfWeek": [pd.Timestamp(future_date).dayofweek],
        "Inventory Level": [inventory],
        "Price": [price],
        "Discount": [discount],
        "Competitor Pricing": [competitor_price],
        "Holiday/Promotion": [holiday]
    })

    future_input = pd.get_dummies(
        future_input,
        columns=["Holiday/Promotion"],
        dtype=int
    )

    future_input = future_input.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    result = model.predict(future_input)[0]

    st.success(
        f"Predicted Units Sold: {result:.2f}"
    )

st.divider()
st.caption("Retail Sales Forecasting using Python, Machine Learning and Streamlit")
