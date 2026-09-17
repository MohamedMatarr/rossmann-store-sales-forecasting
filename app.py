import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date as date_cls

# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Rossmann Sales Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Custom styling
# ------------------------------------------------------------------
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }

    /* Hero banner: fixed gradient + white text, looks the same on any theme */
    .hero {
        background: linear-gradient(135deg, #1f4e8c 0%, #2f7cd6 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: #ffffff !important;
        margin-bottom: 1.5rem;
    }
    .hero h1, .hero p {
        color: #ffffff !important;
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .hero p {
        margin-top: 0.5rem;
        font-size: 1.05rem;
        opacity: 0.92;
    }

    /* Prediction card: use Streamlit's own theme variables so text is
       always readable, whether the app is in Light or Dark mode */
    .metric-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 14px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        text-align: center;
    }
    .metric-card .metric-label {
        color: var(--text-color);
        font-weight: 600;
        opacity: 0.85;
    }
    .metric-card h2 {
        color: #2f7cd6;
        font-size: 2.4rem;
        margin: 0.3rem 0;
    }

    .closed-banner {
        background: rgba(255, 180, 60, 0.15);
        border: 1px solid rgba(255, 180, 60, 0.5);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        color: var(--text-color);
        font-weight: 600;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #1f4e8c 0%, #2f7cd6 100%);
        color: #ffffff !important;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
    }
    div.stButton > button:hover {
        opacity: 0.92;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Hero header
# ------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>📈 Rossmann Store Sales Predictor</h1>
    <p>Forecast expected daily sales for any store using a trained Gradient Boosting model.</p>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("xgb_model.pkl")
    cols = joblib.load("model_columns.pkl")
    return model, cols

try:
    xgb_model, model_columns = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.error(
        "⚠️ Model files not found. Make sure `xgb_model.pkl` and `model_columns.pkl` "
        "are in the same directory as this app."
    )

# ------------------------------------------------------------------
# Sidebar inputs
# ------------------------------------------------------------------
st.sidebar.header("🏬 Store & Date Info")

store = st.sidebar.number_input("Store ID", min_value=1, max_value=1115, value=1, step=1)
input_date = st.sidebar.date_input("Date", value=date_cls.today())
day_of_week = st.sidebar.selectbox(
    "Day of Week", [1, 2, 3, 4, 5, 6, 7],
    format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday",
                            "Friday", "Saturday", "Sunday"][x - 1],
)
store_open = st.sidebar.toggle("Store is open", value=True)
open_status = "Yes" if store_open else "No"

st.sidebar.divider()
st.sidebar.header("🎯 Promotions & Holidays")

promo_on = st.sidebar.toggle("Running promo today", value=False)
promo = "Yes" if promo_on else "No"
school_holiday_on = st.sidebar.toggle("School holiday", value=False)
school_holiday = "Yes" if school_holiday_on else "No"
state_holiday = st.sidebar.selectbox(
    "State holiday", ["None", "Public holiday", "Easter", "Christmas"]
)

st.sidebar.divider()
st.sidebar.header("🏪 Store Profile")

store_type = st.sidebar.selectbox(
    "Store Type", [0, 1, 2, 3], format_func=lambda x: f"Type {chr(65 + x)}"
)
assortment = st.sidebar.selectbox(
    "Assortment Level", [0, 1, 2], format_func=lambda x: f"Level {chr(65 + x)}"
)
competition_dist = st.sidebar.number_input(
    "Distance to nearest competitor (meters)", min_value=0.0, value=1270.0, step=10.0
)

with st.sidebar.expander("Advanced: competition & promo2 history"):
    comp_open_month = st.number_input("Competition open since (month)", 1, 12, 1)
    comp_open_year = st.number_input("Competition open since (year)", 1900, 2020, 2010)
    promo2 = st.radio("Enrolled in Promo2?", ["No", "Yes"], horizontal=True)
    promo2_week = st.number_input("Promo2 since (week)", 0, 52, 0)
    promo2_year = st.number_input("Promo2 since (year)", 0, 2020, 0)

predict_clicked = st.sidebar.button("🚀 Predict Sales")

# ------------------------------------------------------------------
# Main area
# ------------------------------------------------------------------
left, right = st.columns([1.3, 1])

with left:
    st.subheader("Summary of Selected Inputs")
    summary_df = pd.DataFrame({
        "Field": ["Store ID", "Date", "Day of Week", "Open", "Promo",
                  "School Holiday", "State Holiday", "Store Type",
                  "Assortment", "Competitor Distance (m)"],
        "Value": [store, input_date.strftime("%Y-%m-%d"),
                  ["Monday", "Tuesday", "Wednesday", "Thursday",
                   "Friday", "Saturday", "Sunday"][day_of_week - 1],
                  open_status, promo, school_holiday, state_holiday,
                  f"Type {chr(65 + store_type)}", f"Level {chr(65 + assortment)}",
                  competition_dist],
    })
    st.dataframe(summary_df, hide_index=True, use_container_width=True)

with right:
    st.subheader("Prediction")

    if not model_loaded:
        st.info("Load a valid model to see predictions here.")
    elif not predict_clicked:
        st.info("Fill in the store details on the left and click **Predict Sales**.")
    elif open_status == "No":
        st.markdown(
            '<div class="closed-banner">🚫 Store is closed on this day.<br>'
            'Expected sales: <b>€0.00</b></div>',
            unsafe_allow_html=True,
        )
    else:
        state_holiday_map = {"None": 0, "Public holiday": 1, "Easter": 2, "Christmas": 3}

        input_data = pd.DataFrame([{
            "Store": store,
            "DayOfWeek": day_of_week,
            "Promo": 1 if promo == "Yes" else 0,
            "StateHoliday": state_holiday_map[state_holiday],
            "SchoolHoliday": 1 if school_holiday == "Yes" else 0,
            "StoreType": store_type,
            "Assortment": assortment,
            "CompetitionDistance": competition_dist,
            "CompetitionOpenSinceMonth": comp_open_month,
            "CompetitionOpenSinceYear": comp_open_year,
            "Promo2": 1 if promo2 == "Yes" else 0,
            "Promo2SinceWeek": promo2_week,
            "Promo2SinceYear": promo2_year,
            "PromoInterval": 0,
            "Year": input_date.year,
            "Month": input_date.month,
            "Day": input_date.day,
            "WeekOfYear": input_date.isocalendar()[1],
            "Diff_Months": max(
                0,
                (input_date.year - comp_open_year) * 12 + (input_date.month - comp_open_month),
            ),
        }])

        # Keep only the columns the model was trained on, in the right order
        missing_cols = [c for c in model_columns if c not in input_data.columns]
        for c in missing_cols:
            input_data[c] = 0
        input_data = input_data[model_columns]

        prediction = xgb_model.predict(input_data)[0]

        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Expected Sales</div>'
            f'<h2>€{prediction:,.2f}</h2>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            "Estimate produced by a Gradient Boosting Regressor trained on the "
            "Rossmann store sales dataset. Actual results may vary."
        )

st.divider()
with st.expander("ℹ️ About this app"):
    st.write(
        "This app uses a Gradient Boosting model trained on the Rossmann Store "
        "Sales dataset to estimate a store's daily sales based on calendar, "
        "promotion, and store-profile information. It is intended for "
        "demonstration purposes and should not be used for financial decisions."
    )
