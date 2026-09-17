# Rossmann Store Sales Forecasting

Predicting daily sales for over 1,100 Rossmann drug stores across Europe. The project covers data cleaning, feature engineering on time-series data, and a comparison between three boosting models: AdaBoost, Gradient Boosting, and XGBoost.

## Project Overview

- **Goal:** Forecast daily store sales to help with inventory and operational planning.
- **Data:** Rossmann Store Sales dataset (train, test, and store metadata).
- **Metrics:** RMSE and R² on a time-based validation split.

## Data Pipeline

1. **Cleaning**
   - Merged sales data with store metadata.
   - Filled missing values based on the meaning of each field (e.g. stores not enrolled in Promo2, missing competition open dates).
   - Removed closed stores (`Open == 0`) from training, since they add no useful signal.

2. **Feature Engineering**
   - Extracted date-based features: `Year`, `Month`, `Day`, `WeekOfYear`.
   - Calculated how long a competitor had been open (`CompetitionOpenMonths`).
   - Encoded categorical columns with `LabelEncoder`, fitting on train and test combined to avoid unseen-category errors.

3. **Validation**
   - Used a time-based split (last 6 weeks held out) instead of a random split, since a random split would leak future information into training.

## Results

| Model | RMSE (€) | R² Score | Notes |
|---|---|---|---|
| AdaBoost Regressor | 5,403.03 | -2.12 | Performed poorly on this data; the weak learners (stumps) couldn't capture the non-linear patterns, and predictions were unstable. |
| Gradient Boosting | 1,609.04 | 0.72 | Reduced RMSE by about 70% compared to AdaBoost, by fitting sequentially to the residuals. |
| XGBoost (default params) | 2,438.37 | 0.36 | Added as a further comparison against Gradient Boosting. |
| **XGBoost (GridSearchCV tuned)** | — | **0.8536** | Best-performing model overall after hyperparameter tuning with GridSearchCV. |

**Best Parameters (XGBoost, via GridSearchCV):**
```
{
    'gamma': 0,
    'learning_rate': 0.1,
    'max_depth': 7,
    'n_estimators': 300,
    'reg_lambda': 0.1
}
```

## Feature Importance

- `Promo` had the strongest effect on daily sales — running a promotion clearly moved the numbers.
- `CompetitionDistance` and `StoreType` also mattered, tied to a store's baseline revenue.
- Time-based features (`Month`, `WeekOfYear`) picked up seasonal effects, especially around holidays.

## Project Structure

```
rossmann-store-sales-forecasting/
├── Gradient_Boosting_organized.ipynb
├── app.py
├── gb_model.pkl
├── xgb_model.pkl
├── model_columns.pkl
├── final_submission.csv
├── requirements.txt
└── README.md
```

The Rossmann dataset itself (`train.csv`, `test.csv`, `store.csv`) is not included in the repo — it needs to be downloaded separately (see "How to Run" below).

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/rossmann-store-sales-forecasting.git
   cd rossmann-store-sales-forecasting
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the Rossmann dataset files (`train.csv`, `test.csv`, `store.csv`) and place them in a `data_sets/Rossmann/` folder — or update the file paths in the notebook to wherever you keep them.

4. Run the notebook to reproduce the analysis and train the models:
   ```bash
   jupyter notebook Gradient_Boosting_organized.ipynb
   ```

5. To try the trained model through a simple web interface, run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Notes

- The dataset isn't included in this repository due to its size — download it separately and place it in `data_sets/Rossmann/` (or point the notebook to wherever you store it).
- `gb_model.pkl`, `xgb_model.pkl`, and `model_columns.pkl` are already included in this repo, so the Streamlit app works out of the box. If you retrain any model, re-save the relevant files with the same names before running the app.
- `xgboost` has been added to `requirements.txt` to support the new model.
- `final_submission.csv` is the prediction output generated from the test set at the end of the notebook.
