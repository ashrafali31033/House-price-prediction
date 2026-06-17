"""
preprocess.py
-------------
Generates a synthetic housing dataset and performs:
- Data cleaning
- Feature engineering
- Encoding & scaling
- Train/test split
Saves processed data to data/ directory.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# ─── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ─── Generate Synthetic Dataset ──────────────────────────────────────────────
def generate_dataset(n=1000):
    """Create a realistic synthetic housing dataset."""
    area           = np.random.randint(500, 5000, n)
    bedrooms       = np.random.randint(1, 6, n)
    bathrooms      = np.random.randint(1, 4, n)
    floors         = np.random.randint(1, 4, n)
    year_built     = np.random.randint(1970, 2023, n)
    garage         = np.random.randint(0, 2, n)
    location_score = np.round(np.random.uniform(1, 10, n), 1)
    distance_city  = np.round(np.random.uniform(1, 50, n), 1)
    renovated      = np.random.randint(0, 2, n)

    # Price formula with noise
    price = (
        area * 120
        + bedrooms * 8000
        + bathrooms * 5000
        + floors * 3000
        + (2023 - year_built) * (-400)
        + garage * 15000
        + location_score * 12000
        + distance_city * (-2000)
        + renovated * 20000
        + np.random.normal(0, 15000, n)
    )
    price = np.clip(price, 50000, None)  # no negative prices

    df = pd.DataFrame({
        'area':           area,
        'bedrooms':       bedrooms,
        'bathrooms':      bathrooms,
        'floors':         floors,
        'year_built':     year_built,
        'garage':         garage,
        'location_score': location_score,
        'distance_city':  distance_city,
        'renovated':      renovated,
        'price':          price.astype(int),
    })
    return df


# ─── Preprocessing ───────────────────────────────────────────────────────────
def preprocess(df):
    """Clean and engineer features."""
    print("── Raw Data ──────────────────────────────────")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nBasic stats:\n{df.describe()}\n")

    # Feature engineering
    df['house_age']       = 2024 - df['year_built']
    df['area_per_room']   = df['area'] / (df['bedrooms'] + df['bathrooms'])
    df['price_per_sqft']  = df['price'] / df['area']   # kept for analysis, dropped for features

    # Features & target
    feature_cols = [
        'area', 'bedrooms', 'bathrooms', 'floors',
        'house_age', 'garage', 'location_score',
        'distance_city', 'renovated', 'area_per_room'
    ]
    X = df[feature_cols]
    y = df['price']

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
    X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=feature_cols)

    print("── Processed Data ────────────────────────────")
    print(f"Training samples : {len(X_train)}")
    print(f"Testing  samples : {len(X_test)}")
    print(f"Features         : {feature_cols}\n")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    df = generate_dataset(1000)
    df.to_csv("data/housing_data.csv", index=False)
    print("✅ Dataset saved → data/housing_data.csv\n")

    X_train, X_test, y_train, y_test, scaler, features = preprocess(df)
    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv",  index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv",  index=False)
    print("✅ Processed splits saved to data/")
