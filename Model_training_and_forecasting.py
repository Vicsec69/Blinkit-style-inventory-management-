# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error

# # Load feature-engineered data
# df = pd.read_csv(r"C:\Users\shiva\OneDrive\Desktop\Desk\Blinkit-Style Grocery Demand Forecasting System\demand-forecasting-kernels-only/blinkit_features.csv")
# df['date'] = pd.to_datetime(df['date'])

# # -----------------------------
# # TRAIN / TEST SPLIT (TIME-BASED)
# # -----------------------------
# train_size = int(len(df) * 0.8)
# train_df = df.iloc[:train_size]
# test_df = df.iloc[train_size:]

# FEATURES = [
#     'day_of_week',
#     'weekend',
#     'month',
#     'lag_1',
#     'lag_7',
#     'rolling_mean_7',
#     'rolling_mean_14'
# ]

# TARGET = 'units_sold'

# X_train = train_df[FEATURES]
# y_train = train_df[TARGET]

# X_test = test_df[FEATURES]
# y_test = test_df[TARGET]

# # -----------------------------
# # BASELINE MODEL (LAG-1)
# # -----------------------------
# baseline_preds = test_df['lag_1']

# baseline_mae = mean_absolute_error(y_test, baseline_preds)
# baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))

# print("📊 Baseline Model Performance")
# print(f"MAE  : {baseline_mae:.2f}")
# print(f"RMSE : {baseline_rmse:.2f}")

# # -----------------------------
# # LINEAR REGRESSION MODEL
# # -----------------------------
# model = LinearRegression()
# model.fit(X_train, y_train)

# ml_preds = model.predict(X_test)

# ml_mae = mean_absolute_error(y_test, ml_preds)
# ml_rmse = np.sqrt(mean_squared_error(y_test, ml_preds))

# print("\n📊 ML Model (Linear Regression) Performance")
# print(f"MAE  : {ml_mae:.2f}")
# print(f"RMSE : {ml_rmse:.2f}")

# # -----------------------------
# # COMPARE MODELS
# # -----------------------------
# print("\n📈 Model Comparison")
# if ml_rmse < baseline_rmse:
#     print("✅ ML model outperforms baseline")
# else:
#     print("⚠️ Baseline performs better (needs better features/model)")


# # Predict for the next seven days
# last_row = df.iloc[-1:].copy()

# future_preds = []

# for day in range(7):
#     pred = model.predict(last_row[FEATURES])[0]
#     future_preds.append(pred)

#     # Shift features forward
#     last_row['lag_7'] = last_row['lag_1']
#     last_row['lag_1'] = pred
#     last_row['rolling_mean_7'] = (
#         last_row['rolling_mean_7'] * 6 + pred
#     ) / 7

# print("\n📅 Next 7-Day Demand Forecast:")
# print([round(p, 2) for p in future_preds])

# # Visualization
# import matplotlib.pyplot as plt

# FEATURES = [
#     'day_of_week',
#     'weekend',
#     'month',
#     'lag_1',
#     'lag_7',
#     'rolling_mean_7',
#     'rolling_mean_14'
# ]
# TARGET = 'units_sold'

# # Train-test split (time-based)
# train_size = int(len(df) * 0.8)
# train_df = df.iloc[:train_size]
# test_df = df.iloc[train_size:]

# X_train = train_df[FEATURES]
# y_train = train_df[TARGET]
# X_test = test_df[FEATURES]
# y_test = test_df[TARGET]

# # -----------------------------
# # TRAIN MODEL
# # -----------------------------
# model = LinearRegression()
# model.fit(X_train, y_train)

# test_df = test_df.copy()
# test_df['predicted_units'] = model.predict(X_test)

# # -----------------------------
# # PLOT 1: ACTUAL vs PREDICTED
# # -----------------------------
# plt.figure()
# plt.plot(test_df['date'], y_test.values)
# plt.plot(test_df['date'], test_df['predicted_units'].values)
# plt.title("Actual vs Predicted Demand (Blinkit Store)")
# plt.xlabel("Date")
# plt.ylabel("Units Sold")
# plt.legend(["Actual Demand", "Predicted Demand"])
# plt.show()

# # -----------------------------
# # FORECAST NEXT 7 DAYS
# # -----------------------------
# last_row = df.iloc[-1:].copy()
# future_dates = pd.date_range(
#     start=last_row['date'].iloc[0] + pd.Timedelta(days=1),
#     periods=7,
#     freq='D'
# )

# future_predictions = []

# for _ in range(7):
#     prediction = model.predict(last_row[FEATURES])[0]
#     future_predictions.append(prediction)

#     # Shift features forward
#     last_row['lag_7'] = last_row['lag_1']
#     last_row['lag_1'] = prediction
#     last_row['rolling_mean_7'] = (
#         last_row['rolling_mean_7'] * 6 + prediction
#     ) / 7
#     last_row['rolling_mean_14'] = (
#         last_row['rolling_mean_14'] * 13 + prediction
#     ) / 14

# # -----------------------------
# # PLOT 2: FUTURE FORECAST
# # -----------------------------
# plt.figure()

# plt.bar(
#     future_dates.strftime('%a\n%d %b'),
#     future_predictions
# )

# plt.title("7-Day Demand Forecast (Blinkit Store)")
# plt.xlabel("Date")
# plt.ylabel("Forecasted Units Sold")

# # Add value labels on top of bars
# for i, value in enumerate(future_predictions):
#     plt.text(
#         i,
#         value,
#         f"{int(value)}",
#         ha='center',
#         va='bottom'
#     )

# plt.show()

# ==========================================
# BLINKIT-STYLE GROCERY DEMAND FORECASTING
# (Corrected & Production-Grade)
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ------------------------------------------
# LOAD DATA
# ------------------------------------------
df = pd.read_csv(
    r"C:\Users\shiva\OneDrive\Desktop\Desk\Blinkit-Style Grocery Demand Forecasting System\demand-forecasting-kernels-only/blinkit_features.csv"
)

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# ------------------------------------------
# FEATURES & TARGET
# ------------------------------------------
FEATURES = [
    'day_of_week',
    'weekend',
    'month',
    'lag_1',
    'lag_7',
    'rolling_mean_7',
    'rolling_mean_14'
]

TARGET = 'units_sold'

# ------------------------------------------
# TIME-BASED TRAIN / TEST SPLIT
# ------------------------------------------
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

# ------------------------------------------
# BASELINE MODEL (LAG-1)
# ------------------------------------------
baseline_preds = test_df['lag_1']

baseline_mae = mean_absolute_error(y_test, baseline_preds)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))

print("📊 Baseline Model Performance")
print(f"MAE  : {baseline_mae:.2f}")
print(f"RMSE : {baseline_rmse:.2f}")

# ------------------------------------------
# LINEAR REGRESSION MODEL
# ------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

test_preds = model.predict(X_test)

ml_mae = mean_absolute_error(y_test, test_preds)
ml_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

print("\n📊 ML Model (Linear Regression) Performance")
print(f"MAE  : {ml_mae:.2f}")
print(f"RMSE : {ml_rmse:.2f}")

print("\n📈 Model Comparison")
if ml_rmse < baseline_rmse:
    print("✅ ML model outperforms baseline")
else:
    print("⚠️ Baseline performs better (add features or better model)")

# ------------------------------------------
# ACTUAL vs PREDICTED PLOT
# ------------------------------------------
test_df = test_df.copy()
test_df['predicted_units'] = test_preds

plt.figure(figsize=(10, 5))
plt.plot(test_df['date'], y_test, label='Actual Demand')
plt.plot(test_df['date'], test_df['predicted_units'], label='Predicted Demand')
plt.title("Actual vs Predicted Demand (Blinkit Store)")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()
plt.show()

# ------------------------------------------
# 7-DAY FUTURE FORECAST (CORRECT METHOD)
# ------------------------------------------
history = df[TARGET].iloc[-14:].tolist()
last_date = df['date'].iloc[-1]

future_predictions = []
future_dates = []

for i in range(7):
    next_date = last_date + pd.Timedelta(days=i + 1)

    features = {
        'day_of_week': next_date.dayofweek,
        'weekend': 1 if next_date.dayofweek >= 5 else 0,
        'month': next_date.month,
        'lag_1': history[-1],
        'lag_7': history[-7],
        'rolling_mean_7': np.mean(history[-7:]),
        'rolling_mean_14': np.mean(history[-14:])
    }

    X_future = pd.DataFrame([features])
    prediction = model.predict(X_future)[0]

    # Prevent negative demand
    prediction = max(0, prediction)

    future_predictions.append(prediction)
    future_dates.append(next_date)
    history.append(prediction)

print("\n📅 Next 7-Day Demand Forecast:")
print([round(p, 2) for p in future_predictions])

# ------------------------------------------
# FUTURE FORECAST VISUALIZATION
# ------------------------------------------
plt.figure(figsize=(8, 5))
plt.bar(
    [d.strftime('%a\n%d %b') for d in future_dates],
    future_predictions
)

plt.title("7-Day Demand Forecast (Blinkit Store)")
plt.xlabel("Date")
plt.ylabel("Forecasted Units Sold")

for i, value in enumerate(future_predictions):
    plt.text(i, value, f"{int(value)}", ha='center', va='bottom')

plt.show()
