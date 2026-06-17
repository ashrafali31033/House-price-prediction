# 🏠 House Price Prediction — ML Project

> **B.Tech Data Science | Machine Learning Project**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

This project predicts house prices using various Machine Learning regression algorithms. It covers the complete ML pipeline — from data preprocessing and EDA to model training, evaluation, and visualization.

---

## 🎯 Objectives

- Perform Exploratory Data Analysis (EDA) on housing data
- Preprocess and engineer features for better model accuracy
- Train and compare multiple ML regression models
- Evaluate models using standard metrics (RMSE, MAE, R²)
- Visualize results and feature importances

---

## 🗂️ Project Structure

```
house-price-prediction/
│
├── data/
│   └── housing_data.csv          # Dataset (auto-generated)
│
├── scripts/
│   ├── preprocess.py             # Data cleaning & feature engineering
│   ├── train.py                  # Model training & evaluation
│   └── visualize.py              # Plotting & visualization
│
├── visualizations/               # Saved plots (auto-generated)
│
├── report/
│   └── project_report.md         # Detailed project report
│
├── House_Price_Prediction.ipynb  # Main Jupyter Notebook
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 🤖 Models Used

| Model | Description |
|---|---|
| Linear Regression | Baseline regression model |
| Ridge Regression | L2 regularized linear model |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Boosted ensemble model |
| XGBoost | Optimized gradient boosting |

---

## 📊 Features Used

| Feature | Description |
|---|---|
| `area` | Total area in sq ft |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `floors` | Number of floors |
| `year_built` | Year of construction |
| `garage` | Garage availability (0/1) |
| `location_score` | Neighborhood quality score |
| `distance_city` | Distance from city center (km) |
| `renovated` | Whether renovated (0/1) |

---

## ⚙️ Setup & Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Python Scripts
```bash
# Preprocess data
python scripts/preprocess.py

# Train models
python scripts/train.py

# Generate visualizations
python scripts/visualize.py
```

### 4. Or Open the Jupyter Notebook
```bash
jupyter notebook House_Price_Prediction.ipynb
```

---

## 📈 Results

| Model | RMSE | MAE | R² Score |
|---|---|---|---|
| Linear Regression | ~28,500 | ~21,000 | ~0.81 |
| Ridge Regression | ~27,900 | ~20,500 | ~0.82 |
| Random Forest | ~19,200 | ~14,300 | ~0.91 |
| Gradient Boosting | ~17,800 | ~13,100 | ~0.93 |
| XGBoost | ~16,500 | ~12,200 | ~0.94 |

> *Results may vary slightly due to random seed differences.*

---

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Kaggle House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

---

## 👤 Author

**Your Name**  
B.Tech Data Science | [Your College Name]  
[your.email@example.com]

---

## 📄 License

This project is licensed under the MIT License.
