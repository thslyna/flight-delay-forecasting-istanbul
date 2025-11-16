import os
import pandas as pd

# === Load merged data ===
p = os.path.expanduser('~/thesis/data/processed/training_data.csv')
out = os.path.expanduser('~/thesis/data/processed/training_features.csv')

df = pd.read_csv(p, parse_dates=['hour'])

print("📦 Loaded merged dataset:", df.shape)

# === Step 1: Time-based features ===
df['hour_of_day'] = df['hour'].dt.hour
df['day_of_week'] = df['hour'].dt.dayofweek  # 0 = Monday, 6 = Sunday
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# === Step 2: Handle missing values ===
for col in ['temp', 'humidity', 'pressure', 'wind_speed', 'avg_speed', 'total_vehicles']:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# === Step 3: Simplify weather categories ===
if 'weather' in df.columns:
    df['weather'] = df['weather'].fillna('Unknown')
    common_weather = df['weather'].value_counts().index[:5]
    df['weather'] = df['weather'].apply(lambda x: x if x in common_weather else 'Other')

# === Step 4: Target variable ===
df['delayed_15'] = (df['delay_min'] >= 15).astype(int)

# === Step 5: Save final feature dataset ===
df.to_csv(out, index=False)
print(f"✅ Features saved to {out} with shape {df.shape}")
print("📊 Columns:", df.columns.tolist())
