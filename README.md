# Rossmann Store Sales Forecasting

Predicting daily sales for over 1,100 Rossmann drug stores across Europe. The project covers data cleaning, feature engineering on time-series data, and a comparison between two boosting models: AdaBoost and Gradient Boosting.

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
| Gradient Boosting | **1,609.04** | **0.72** | Reduced RMSE by about 70% compared to AdaBoost, by fitting sequentially to the residuals. |

## Feature Importance

- `Promo` had the strongest effect on daily sales — running a promotion clearly moved the numbers.
- `CompetitionDistance` and `StoreType` also mattered, tied to a store's baseline revenue.
- Time-based features (`Month`, `WeekOfYear`) picked up seasonal effects, especially around holidays.

## Project Structure

```
rossmann-store-sales-forecasting/
├── data_sets/
│   └── Rossmann/
│       ├── train.csv
│       ├── test.csv
│       └── store.csv
├── notebooks/
│   └── Gradient_Boosting_organized.ipynb
├── app.py
├── requirements.txt
└── README.md
```

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

3. Add the Rossmann dataset files (`train.csv`, `test.csv`, `store.csv`) under `data_sets/Rossmann/`.

4. Run the notebook to reproduce the analysis and train the models:
   ```bash
   jupyter notebook notebooks/Gradient_Boosting_organized.ipynb
   ```

5. To try the trained model through a simple web interface, run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Notes

- The dataset isn't included in this repository due to its size — download it separately and place it in `data_sets/Rossmann/`.
- The trained model (`gb_model.pkl`) and its expected column order (`model_columns.pkl`) need to be saved from the notebook before the Streamlit app can produce predictions.
