import pandas as pd

# Load the dataset
df = pd.read_csv(r"C:\Users\shiva\OneDrive\Desktop\Desk\Blinkit-Style Grocery Demand Forecasting System\demand-forecasting-kernels-only\train.csv")

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# -----------------------------
# BUSINESS SCOPE DEFINITION
# -----------------------------

# Choose one Blinkit dark store
STORE_ID = 1

# Choose top 5 fast-moving items in that store
top_items = (
    df[df['store'] == STORE_ID]
    .groupby('item')['sales']
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .index
)

# Filter data for selected store and items
filtered_df = df[
    (df['store'] == STORE_ID) &
    (df['item'].isin(top_items))
].copy()

# Sort for time-series correctness
filtered_df = filtered_df.sort_values(by=['item', 'date'])

# Rename columns to business-friendly names
filtered_df.rename(
    columns={
        'store': 'store_id',
        'item': 'product_id',
        'sales': 'units_sold'
    },
    inplace=True
)

# Save processed dataset
filtered_df.to_csv(r"C:\Users\shiva\OneDrive\Desktop\Desk\Blinkit-Style Grocery Demand Forecasting System\demand-forecasting-kernels-only\blinkit_store_scope.csv", index=False)



# Data Cleaning 

df = pd.read_csv(r"C:\Users\shiva\OneDrive\Desktop\Desk\Blinkit-Style Grocery Demand Forecasting System\demand-forecasting-kernels-only\blinkit_store_scope.csv")
df['date'] = pd.to_datetime(df['date'])

# -----------------------------
# HANDLE MISSING DATES
# -----------------------------

cleaned_data = []

# Process each product separately
for product_id in df['product_id'].unique():
    product_df = df[df['product_id'] == product_id].copy()

    # Create complete daily date range
    full_dates = pd.date_range(
        start=product_df['date'].min(),
        end=product_df['date'].max(),
        freq='D'
    )

    # Reindex to full date range
    product_df = (
        product_df
        .set_index('date')
        .reindex(full_dates)
        .rename_axis('date')
        .reset_index()
    )

    # Fill missing values
    product_df['units_sold'] = product_df['units_sold'].fillna(0)
    product_df['product_id'] = product_id
    product_df['store_id'] = product_df['store_id'].fillna(method='ffill')

    cleaned_data.append(product_df)

# Combine all products
cleaned_df = pd.concat(cleaned_data, ignore_index=True)

# Sort final data
cleaned_df = cleaned_df.sort_values(by=['product_id', 'date'])

# Save cleaned dataset
cleaned_df.to_csv(
    r"C:\Users\shiva\OneDrive\Desktop\Desk\Blinkit-Style Grocery Demand Forecasting System\demand-forecasting-kernels-only/blinkit_cleaned_timeseries.csv",
    index=False
)

print(f"Store selected: {STORE_ID}")
print(f"Products selected: {list(top_items)}")
print(f"Final rows: {filtered_df.shape[0]}")
print(f"Final dataset shape: {cleaned_df.shape}")
