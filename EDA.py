import matplotlib.pyplot as plt
import pandas as pd

# Load cleaned time-series data
df = pd.read_csv(r"C:\Users\shiva\OneDrive\Desktop\Desk\Blinkit-Style Grocery Demand Forecasting System\demand-forecasting-kernels-only/blinkit_cleaned_timeseries.csv")
df['date'] = pd.to_datetime(df['date'])



daily_sales = df.groupby('date')['units_sold'].sum()


# -----------------------------
# 7-DAY ROLLING AVERAGE (SEASONALITY)
# -----------------------------
rolling_sales = daily_sales.rolling(window=7).mean()

plt.figure()
plt.plot(daily_sales.index, daily_sales.values, label='Daily Sales')
plt.plot(rolling_sales.index, rolling_sales.values, label='7-Day Rolling Avg')
plt.title("Daily Demand with 7-Day Rolling Average")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()
plt.show()

# -----------------------------
# WEEKDAY ANALYSIS
# -----------------------------
df['weekday'] = df['date'].dt.day_name()

weekday_sales = (
    df.groupby('weekday')['units_sold']
    .mean()
    .reindex([
        'Monday', 'Tuesday', 'Wednesday',
        'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])
)

plt.figure()
plt.bar(weekday_sales.index, weekday_sales.values)
plt.title("Average Demand by Weekday")
plt.xlabel("Weekday")
plt.ylabel("Average Units Sold")
plt.xticks(rotation=45)
plt.show()

# Feature Engineering

df = df.sort_values(by=['product_id', 'date'])

# -----------------------------
# TIME-BASED FEATURES
# -----------------------------
df['day_of_week'] = df['date'].dt.dayofweek   # 0=Monday
df['weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['month'] = df['date'].dt.month

# -----------------------------
# LAG FEATURES
# -----------------------------
df['lag_1'] = df.groupby('product_id')['units_sold'].shift(1)
df['lag_7'] = df.groupby('product_id')['units_sold'].shift(7)

# -----------------------------
# ROLLING WINDOW FEATURES
# -----------------------------
df['rolling_mean_7'] = (
    df.groupby('product_id')['units_sold']
    .shift(1)
    .rolling(window=7)
    .mean()
)

df['rolling_mean_14'] = (
    df.groupby('product_id')['units_sold']
    .shift(1)
    .rolling(window=14)
    .mean()
)

# -----------------------------
# HANDLE MISSING VALUES
# -----------------------------
df = df.dropna().reset_index(drop=True)

# Save feature-engineered dataset
df.to_csv(
    r"C:\Users\shiva\OneDrive\Desktop\Desk\Blinkit-Style Grocery Demand Forecasting System\demand-forecasting-kernels-only/blinkit_features.csv",
    index=False
)


print("Final feature dataset shape:", df.shape)

# Model training and Forecasting 

df['month'] = df['date'].dt.to_period('M')
monthly_avg = df.groupby('month')['units_sold'].mean()

plt.figure()
monthly_avg.plot(kind='bar')
plt.title("Average Monthly Demand (Blinkit Store)")
plt.xlabel("Month")
plt.ylabel("Average Units Sold")
plt.show()

plt.figure()

df['week'] = df['date'].dt.to_period('W')

# Weekly average demand per product
weekly_avg = (
    df.groupby(['week', 'product_id'])['units_sold']
    .mean()
    .reset_index()
)

# Convert week back to timestamp for plotting
weekly_avg['week'] = weekly_avg['week'].dt.start_time

for product_id in weekly_avg['product_id'].unique():
    product_data = weekly_avg[weekly_avg['product_id'] == product_id]
    plt.plot(
        product_data['week'],
        product_data['units_sold'],
        label=f'Product {product_id}'
    )

plt.title("Weekly Average Demand by Product")
plt.xlabel("Week")
plt.ylabel("Average Units Sold")
plt.legend()
plt.show()