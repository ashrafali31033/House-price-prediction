"""
visualize.py
------------
Generates and saves all project visualizations:
  1. Price distribution
  2. Correlation heatmap
  3. Feature vs Price scatter plots
  4. Model comparison bar chart
  5. Actual vs Predicted (best model)
  6. Feature importance (Random Forest)
  7. Residual plot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
from sklearn.ensemble  import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from xgboost           import XGBRegressor
from sklearn.metrics   import mean_squared_error, r2_score

from scripts.preprocess import generate_dataset, preprocess

# ─── Style ───────────────────────────────────────────────────────────────────
PALETTE   = "#2563EB"
ACCENT    = "#F59E0B"
BG        = "#F8FAFC"
GRID      = "#E2E8F0"
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "axes.edgecolor":   GRID,
    "grid.color":       GRID,
    "font.family":      "DejaVu Sans",
})

SEED = 42
os.makedirs("visualizations", exist_ok=True)


# ─── 1. Price Distribution ────────────────────────────────────────────────────
def plot_price_dist(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("House Price Distribution", fontsize=14, fontweight="bold")

    axes[0].hist(df["price"], bins=40, color=PALETTE, edgecolor="white", linewidth=0.5)
    axes[0].set_title("Raw Price")
    axes[0].set_xlabel("Price (₹)")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1e5:.0f}L"))

    axes[1].hist(np.log1p(df["price"]), bins=40, color=ACCENT, edgecolor="white", linewidth=0.5)
    axes[1].set_title("Log-Transformed Price")
    axes[1].set_xlabel("log(Price)")

    plt.tight_layout()
    plt.savefig("visualizations/01_price_distribution.png", dpi=150)
    plt.close()
    print("✅ 01_price_distribution.png")


# ─── 2. Correlation Heatmap ───────────────────────────────────────────────────
def plot_correlation(df):
    corr = df[["area","bedrooms","bathrooms","floors","year_built",
               "garage","location_score","distance_city","renovated","price"]].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                square=True, linewidths=0.5, ax=ax,
                annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("visualizations/02_correlation_heatmap.png", dpi=150)
    plt.close()
    print("✅ 02_correlation_heatmap.png")


# ─── 3. Scatter — Key Features vs Price ──────────────────────────────────────
def plot_scatter(df):
    features = ["area", "location_score", "distance_city", "bedrooms"]
    labels   = ["Area (sq ft)", "Location Score", "Distance from City (km)", "Bedrooms"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Key Features vs House Price", fontsize=14, fontweight="bold")

    for ax, feat, label in zip(axes.flat, features, labels):
        ax.scatter(df[feat], df["price"], alpha=0.35, s=15, color=PALETTE)
        ax.set_xlabel(label)
        ax.set_ylabel("Price (₹)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1e5:.0f}L"))

    plt.tight_layout()
    plt.savefig("visualizations/03_feature_vs_price.png", dpi=150)
    plt.close()
    print("✅ 03_feature_vs_price.png")


# ─── 4. Model Comparison Bar Chart ───────────────────────────────────────────
def plot_model_comparison(results):
    models = list(results.keys())
    r2s    = [results[m]["R2"]   for m in models]
    rmses  = [results[m]["RMSE"] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")

    colors = [ACCENT if r == max(r2s) else PALETTE for r in r2s]
    axes[0].barh(models, r2s, color=colors, edgecolor="white")
    axes[0].set_xlabel("R² Score")
    axes[0].set_title("R² Score (higher = better)")
    axes[0].set_xlim(0, 1)
    for i, v in enumerate(r2s):
        axes[0].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)

    colors2 = [ACCENT if r == min(rmses) else PALETTE for r in rmses]
    axes[1].barh(models, rmses, color=colors2, edgecolor="white")
    axes[1].set_xlabel("RMSE (₹)")
    axes[1].set_title("RMSE (lower = better)")
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1e3:.0f}K"))
    for i, v in enumerate(rmses):
        axes[1].text(v + 200, i, f"₹{v/1e3:.1f}K", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("visualizations/04_model_comparison.png", dpi=150)
    plt.close()
    print("✅ 04_model_comparison.png")


# ─── 5. Actual vs Predicted ───────────────────────────────────────────────────
def plot_actual_vs_predicted(y_test, preds, model_name):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, preds, alpha=0.4, s=18, color=PALETTE, label="Predictions")
    mn = min(y_test.min(), preds.min())
    mx = max(y_test.max(), preds.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect Prediction")
    ax.set_xlabel("Actual Price (₹)")
    ax.set_ylabel("Predicted Price (₹)")
    ax.set_title(f"Actual vs Predicted — {model_name}", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1e5:.0f}L"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1e5:.0f}L"))
    r2 = r2_score(y_test, preds)
    ax.legend(title=f"R² = {r2:.4f}")
    plt.tight_layout()
    plt.savefig("visualizations/05_actual_vs_predicted.png", dpi=150)
    plt.close()
    print("✅ 05_actual_vs_predicted.png")


# ─── 6. Feature Importance ────────────────────────────────────────────────────
def plot_feature_importance(model, feature_cols):
    importance = model.feature_importances_
    fi_df = pd.DataFrame({"Feature": feature_cols, "Importance": importance})
    fi_df.sort_values("Importance", ascending=True, inplace=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [ACCENT if v == fi_df["Importance"].max() else PALETTE
              for v in fi_df["Importance"]]
    ax.barh(fi_df["Feature"], fi_df["Importance"], color=colors, edgecolor="white")
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance — Random Forest", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("visualizations/06_feature_importance.png", dpi=150)
    plt.close()
    print("✅ 06_feature_importance.png")


# ─── 7. Residual Plot ─────────────────────────────────────────────────────────
def plot_residuals(y_test, preds, model_name):
    residuals = np.array(y_test) - preds
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Residual Analysis — {model_name}", fontsize=13, fontweight="bold")

    axes[0].scatter(preds, residuals, alpha=0.35, s=15, color=PALETTE)
    axes[0].axhline(0, color="red", lw=1.5, linestyle="--")
    axes[0].set_xlabel("Predicted Price")
    axes[0].set_ylabel("Residuals (₹)")
    axes[0].set_title("Residuals vs Predicted")

    axes[1].hist(residuals, bins=35, color=ACCENT, edgecolor="white", linewidth=0.5)
    axes[1].set_xlabel("Residual Value (₹)")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    plt.savefig("visualizations/07_residual_plot.png", dpi=150)
    plt.close()
    print("✅ 07_residual_plot.png")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = generate_dataset(1000)
    X_train, X_test, y_train, y_test, scaler, features = preprocess(df)

    models = {
        "Linear Regression":  LinearRegression(),
        "Ridge Regression":   Ridge(alpha=1.0),
        "Random Forest":      RandomForestRegressor(n_estimators=100, random_state=SEED),
        "Gradient Boosting":  GradientBoostingRegressor(n_estimators=100, random_state=SEED),
        "XGBoost":            XGBRegressor(n_estimators=100, random_state=SEED,
                                            eval_metric="rmse", verbosity=0),
    }

    results = {}
    best_preds, best_rf = None, None
    for name, m in models.items():
        m.fit(X_train, y_train)
        p = m.predict(X_test)
        results[name] = {
            "R2":   r2_score(y_test, p),
            "RMSE": np.sqrt(mean_squared_error(y_test, p)),
        }
        if name == "XGBoost":
            best_preds = p
        if name == "Random Forest":
            best_rf = m

    print("\n── Generating visualizations ──")
    plot_price_dist(df)
    plot_correlation(df)
    plot_scatter(df)
    plot_model_comparison(results)
    plot_actual_vs_predicted(y_test, best_preds, "XGBoost")
    plot_feature_importance(best_rf, features)
    plot_residuals(y_test, best_preds, "XGBoost")
    print("\n✅ All visualizations saved → visualizations/")
