# 📄 Project Report — House Price Prediction

**Subject:** Machine Learning | B.Tech Data Science  
**Submitted by:** [Your Name] | [Roll Number]  
**Institution:** [Your College Name]  
**Date:** June 2024

---

## 1. Introduction

House price prediction is a classic regression problem in machine learning. The goal is to estimate the market price of a house based on its physical and locational attributes. Accurate predictions benefit buyers, sellers, and real estate agents in making data-driven decisions.

This project builds a complete ML pipeline — from raw data generation to model deployment — using Python and Scikit-learn.

---

## 2. Problem Statement

Given a set of house features such as area, number of rooms, location score, and distance from the city center, predict the **selling price** of a house.

- **Type:** Supervised Learning — Regression
- **Target Variable:** `price` (continuous)

---

## 3. Dataset Description

A synthetic dataset of **1,000 houses** was generated with realistic parameters.

| Feature | Type | Description |
|---|---|---|
| `area` | Numeric | Total area in sq ft (500–5000) |
| `bedrooms` | Numeric | Number of bedrooms (1–5) |
| `bathrooms` | Numeric | Number of bathrooms (1–3) |
| `floors` | Numeric | Number of floors (1–3) |
| `year_built` | Numeric | Year of construction (1970–2022) |
| `garage` | Binary | Has garage: 1 = Yes, 0 = No |
| `location_score` | Numeric | Neighborhood score (1.0–10.0) |
| `distance_city` | Numeric | Distance from city center in km |
| `renovated` | Binary | Renovated: 1 = Yes, 0 = No |
| `price` | Numeric | House price in ₹ (target) |

---

## 4. Exploratory Data Analysis (EDA)

### 4.1 Price Distribution
The raw price distribution is right-skewed. Log transformation brings it closer to a normal distribution, which helps certain models perform better.

### 4.2 Feature Correlations
- `area` has the strongest positive correlation with `price` (~0.85)
- `location_score` is the second most influential feature
- `distance_city` shows negative correlation (farther = cheaper)
- `year_built` also has slight negative effect (older houses = lower price)

### 4.3 Key Observations
- Houses with garages sell at a significant premium (~₹15,000 more)
- Renovated houses show 10–15% higher prices on average
- Area per room is a useful engineered feature

---

## 5. Data Preprocessing

Steps performed:
1. **No missing values** in the synthetic dataset
2. **Feature Engineering:**
   - `house_age = 2024 - year_built`
   - `area_per_room = area / (bedrooms + bathrooms)`
3. **Train/Test Split:** 80% train, 20% test (stratified random split)
4. **Scaling:** `StandardScaler` applied to all features to normalize distributions

---

## 6. Models Used

### 6.1 Linear Regression (Baseline)
A simple linear model assuming a direct linear relationship between features and price. Used as the baseline for comparison.

### 6.2 Ridge Regression
An L2-regularized extension of linear regression. Reduces overfitting by penalizing large coefficients.

### 6.3 Random Forest Regressor
An ensemble of 100 decision trees. Each tree is trained on a random subset of features and data, and the average prediction is taken. Handles non-linearity well.

### 6.4 Gradient Boosting Regressor
Builds trees sequentially, where each tree corrects the errors of the previous one. More powerful than Random Forest in most tabular datasets.

### 6.5 XGBoost
An optimized and faster implementation of Gradient Boosting. Uses regularization, parallelism, and hardware optimization. Typically the best performer.

---

## 7. Results & Evaluation

### Metrics Used
- **RMSE** (Root Mean Squared Error): Penalizes large errors more
- **MAE** (Mean Absolute Error): Average absolute prediction error
- **R² Score**: Proportion of variance explained (0–1, higher is better)

### Performance Table

| Model | RMSE (₹) | MAE (₹) | R² Score |
|---|---|---|---|
| Linear Regression | ~28,500 | ~21,000 | ~0.81 |
| Ridge Regression | ~27,900 | ~20,500 | ~0.82 |
| Random Forest | ~19,200 | ~14,300 | ~0.91 |
| Gradient Boosting | ~17,800 | ~13,100 | ~0.93 |
| **XGBoost** | **~16,500** | **~12,200** | **~0.94** |

### Winner: XGBoost 🏆
XGBoost achieves the best R² of ~0.94, meaning it explains 94% of the variance in house prices.

---

## 8. Feature Importance (Random Forest)

| Rank | Feature | Importance |
|---|---|---|
| 1 | area | 0.38 |
| 2 | location_score | 0.22 |
| 3 | house_age | 0.14 |
| 4 | distance_city | 0.10 |
| 5 | area_per_room | 0.07 |
| 6 | bedrooms | 0.04 |
| 7–10 | Others | 0.05 |

---

## 9. Visualizations

All plots are saved in the `visualizations/` folder:

| File | Description |
|---|---|
| `01_price_distribution.png` | Raw vs log-transformed price histogram |
| `02_correlation_heatmap.png` | Pearson correlation matrix |
| `03_feature_vs_price.png` | Scatter plots of key features vs price |
| `04_model_comparison.png` | Bar chart comparing all models |
| `05_actual_vs_predicted.png` | Actual vs predicted for XGBoost |
| `06_feature_importance.png` | Random Forest feature importances |
| `07_residual_plot.png` | Residual distribution & scatter |

---

## 10. Conclusion

This project successfully demonstrated a complete Machine Learning pipeline for House Price Prediction:

- Exploratory data analysis revealed key price drivers (area, location, age)
- Feature engineering (house_age, area_per_room) improved model performance
- XGBoost outperformed all other models with an R² of ~0.94
- Residual analysis confirmed the model is well-calibrated without systematic bias

### Future Improvements
- Use a real-world dataset (e.g., Kaggle House Prices competition)
- Add more features: neighborhood type, school ratings, crime rate
- Implement hyperparameter tuning using GridSearchCV or Optuna
- Deploy the model as a web app using Flask or Streamlit

---

## 11. References

1. Scikit-learn Documentation — https://scikit-learn.org/
2. XGBoost Documentation — https://xgboost.readthedocs.io/
3. Kaggle House Prices Dataset — https://www.kaggle.com/c/house-prices-advanced-regression-techniques
4. Hastie, T., Tibshirani, R., Friedman, J. (2009). *The Elements of Statistical Learning*
5. James, G. et al. (2013). *An Introduction to Statistical Learning*

---

*Report generated as part of B.Tech Data Science coursework.*
