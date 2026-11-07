
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the dataset
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(BASE_DIR, "retail_store_inventory.csv"))


print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nFirst five rows:")
print(df.head())

print("\nStatistical summary:")
print(df.describe())

print("Dataset loaded successfully!")
print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

# 2. Convert the Date column
df["Date"] = pd.to_datetime(
    df["Date"], format="%d-%m-%Y"
)

# 3. Calculate daily total sales
daily_sales = df.groupby("Date")["Units Sold"].sum()

print("\nDaily sales:")
print(daily_sales.head())


# 4. Calculate monthly total sales
monthly_sales = df.groupby(
    df["Date"].dt.to_period("M")
)["Units Sold"].sum()

# 5. Create a monthly bar graph
plt.figure(figsize=(14, 6))

plt.bar(
    monthly_sales.index.astype(str),
    monthly_sales.values,
    color="royalblue",
    edgecolor="navy",
    width=0.7
)

# 6. Customize the graph
plt.title(
    "Retail Sales Forecasting - Monthly Sales",
    fontsize=17,
    fontweight="bold"
)

plt.xlabel("Month", fontsize=12)
plt.ylabel("Total Units Sold", fontsize=12)

plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

# 7. Save and display
plt.savefig(
    "retail_sales_bar_graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nMonthly sales graph saved successfully!")

# 8. Category-wise Sales Analysis

category_sales = df.groupby("Category")["Units Sold"].sum()

print("\nCategory-wise Sales:")
print(category_sales)

# Create category sales graph
plt.figure(figsize=(10, 6))

plt.bar(
    category_sales.index,
    category_sales.values,
    color="steelblue",
    edgecolor="black"
)

plt.title(
    "Retail Sales by Product Category",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Product Category")
plt.ylabel("Total Units Sold")

plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()

plt.savefig("category_sales.png", dpi=300)
plt.show()

print("\nCategory sales graph saved successfully!")

# 9. Region-wise Sales Analysis

region_sales = df.groupby("Region")["Units Sold"].sum()

print("\nRegion-wise Sales:")
print(region_sales)

# Create region-wise sales graph
plt.figure(figsize=(10, 6))

plt.bar(
    region_sales.index,
    region_sales.values,
    color="cornflowerblue",
    edgecolor="black"
)

plt.title(
    "Retail Sales by Region",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Region")
plt.ylabel("Total Units Sold")

plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()

plt.savefig("region_sales.png", dpi=300)
plt.show()

print("\nRegion-wise sales graph saved successfully!")


# 10. Seasonal Sales Analysis

print("\nAvailable Seasons:")
print(df["Seasonality"].unique())

# Calculate seasonal sales
seasonal_sales = df.groupby("Seasonality")["Units Sold"].sum()

print("\nSeasonal Sales Analysis:")
print(seasonal_sales)

# Create seasonal bar graph
plt.figure(figsize=(10, 6))

plt.bar(
    seasonal_sales.index,
    seasonal_sales.values,
    color=["orange", "royalblue", "green", "purple"],
    edgecolor="black",
    width=0.6
)

plt.title(
    "Retail Sales Forecasting - Seasonal Sales",
    fontsize=17,
    fontweight="bold"
)

plt.xlabel("Season", fontsize=12)
plt.ylabel("Total Units Sold", fontsize=12)

plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

plt.savefig(
    "seasonal_sales.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nSeasonal sales graph saved successfully!")

# 
# 11. Improved Machine Learning Model

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Prepare date features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek

# Select features
features = [
    "Year",
    "Month",
    "Day",
    "DayOfWeek",
    "Inventory Level",
    "Price",
    "Discount",
    "Competitor Pricing",
    "Holiday/Promotion"
]

X = df[features].copy()
y = df["Units Sold"]

# Convert categorical data into numerical data
X = pd.get_dummies(
    X,
    columns=["Holiday/Promotion"],
    dtype=int
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest model
model = RandomForestRegressor(
    n_estimators=150,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predict sales
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n--- Improved Model Evaluation ---")
print("Mean Absolute Error:", round(mae, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R2 Score:", round(r2, 3))

# Actual vs Predicted graph
plt.figure(figsize=(12, 6))

plt.plot(
    range(len(y_test)),
    y_test.values,
    label="Actual Sales",
    color="blue"
)

plt.plot(
    range(len(predictions)),
    predictions,
    label="Predicted Sales",
    color="orange",
    linestyle="--"
)

plt.title("Actual vs Predicted Retail Sales")
plt.xlabel("Test Data Records")
plt.ylabel("Units Sold")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()

plt.savefig("improved_model_graph.png", dpi=300)
plt.show()

print("\nImproved model graph saved successfully!")
## 13. Feature Importance Analysis

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)

# Create feature importance graph
plt.figure(figsize=(12, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"],
    color="teal",
    edgecolor="black"
)

plt.title(
    "Retail Sales Forecasting - Feature Importance",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.gca().invert_yaxis()
plt.grid(axis="x", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.show()

print("\nFeature importance graph saved successfully!")