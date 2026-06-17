"""
train.py
--------
Trains and evaluates five regression models:
  - Linear Regression
  - Ridge Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost

Prints metrics table and saves the best model.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.linear_model    import LinearRegression, Ridge
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score
from xgboost                 import XGBRegressor
from scripts.preprocess      import generate_dataset, preprocess

# ─── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42


# ─── Model Definitions ───────────────────────────────────────────────────────
MODELS = {
    "Linear Regression":    LinearRegression(),
    "Ridge Regression":     Ridge(alpha=1.0),
    "Random Forest":        RandomForestRegressor(n_estimators=100, random_state=SEED),
    "Gradient Boosting":    GradientBoostingRegressor(n_estimators=100, random_state=SEED),
    "XGBoost":              XGBRegressor(n_estimators=100, random_state=SEED,
                                         eval_metric='rmse', verbosity=0),
}


# ─── Evaluation Helper ───────────────────────────────────────────────────────
def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    return rmse, mae, r2, preds


# ─── Training Loop ───────────────────────────────────────────────────────────
def train_all(X_train, X_test, y_train, y_test):
    results = {}
    trained = {}

    print("\n" + "═" * 65)
    print(f"{'Model':<25} {'RMSE':>10} {'MAE':>10} {'R² Score':>10}")
    print("─" * 65)

    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        rmse, mae, r2, preds = evaluate(model, X_test, y_test)
        results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
        trained[name] = (model, preds)
        print(f"{name:<25} {rmse:>10,.0f} {mae:>10,.0f} {r2:>10.4f}")

    print("═" * 65)

    # Best model by R²
    best_name = max(results, key=lambda k: results[k]["R2"])
    print(f"\n🏆 Best Model : {best_name}  (R² = {results[best_name]['R2']:.4f})\n")

    return results, trained, best_name


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = generate_dataset(1000)
    X_train, X_test, y_train, y_test, scaler, features = preprocess(df)

    results, trained_models, best_name = train_all(X_train, X_test, y_train, y_test)

    best_model, _ = trained_models[best_name]
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(scaler,     "models/scaler.pkl")
    print(f"✅ Best model saved → models/best_model.pkl")

    # Save results table
    results_df = pd.DataFrame(results).T.reset_index()
    results_df.columns = ["Model", "RMSE", "MAE", "R2"]
    results_df.to_csv("data/model_results.csv", index=False)
    print("✅ Results saved  → data/model_results.csv")

    return_val = (results, trained_models, X_test, y_test, features)
