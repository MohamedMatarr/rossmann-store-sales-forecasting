# 📈 Rossmann Store Sales Forecasting (AdaBoost vs. Gradient Boosting)

An end-to-end machine learning project predicting daily sales for over 1,100 Rossmann drug stores across Europe. This project explores real-world business constraints, feature engineering on time-series data, and a practical comparative evaluation between **AdaBoost** and **Gradient Boosting**.

---

## 📌 Project Overview
- **Goal:** Accurately forecast daily store `Sales` to optimize inventory and operational planning.
- **Data Source:** Rossmann Store Sales Dataset.
- **Primary Metric:** RMSE and R² Score on a time-based validation set.

---

## 🛠️ Data Pipeline & Methodology
1. **Data Preprocessing & Cleaning:**
   - Merged sales transactions with store metadata.
   - Handled missing values based on domain logic (e.g., non-participating stores in `Promo2`, missing competition dates).
   - Filtered out closed stores (`Open == 0`) and zero-sales records during training to prevent noise.

2. **Feature Engineering:**
   - Extracted temporal features (`Year`, `Month`, `Day`, `WeekOfYear`) from transaction dates.
   - Calculated competition duration in months (`CompetitionOpenMonths`).
   - Applied `LabelEncoder` safely across combined feature categories to eliminate data leakage.

3. **Validation Strategy:**
   - Used a **Time-Based Split** (holding out the last 6 weeks for validation) instead of a random split to prevent future data leakage.

---

## 📊 Model Comparison & Results

| Model | RMSE (€) | R² Score | Key Observations |
| :--- | :---: | :---: | :--- |
| **AdaBoost Regressor** | 5,403.029 | -2.12 | Struggled with non-linear interactions and exhibited extreme sensitivity to outliers due to weak base estimators (stumps). |
| **Gradient Boosting** | **1,609.04** | **0.7226** | **Outperformed AdaBoost by reducing RMSE by over 56%.** Successfully learned non-linear feature interactions via sequential residual learning. |

---

## 🔍 Business Insights (Top Feature Importances)
- **Promo:** Promotional active days proved to be the strongest immediate driver of daily sales volume.
- **Competition Distance & Store Type:** Proximity to competitors and physical store footprint significantly impacted baseline store revenue.
- **Seasonality (Week of Year / Month):** Strong temporal variations were captured during holiday seasons and mid-year periods.

---

## 💻 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/rossmann-store-sales-forecasting.git](https://github.com/YOUR_USERNAME/rossmann-store-sales-forecasting.git)
   cd rossmann-store-sales-forecasting
