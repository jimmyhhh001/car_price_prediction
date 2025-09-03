import os
import io
import pickle
import pandas as pd
import streamlit as st

# Page Config
st.set_page_config(page_title="Car Price Prediction", layout="wide")


st.image("https://motogazer.com/wp-content/uploads/2025/02/MotoGazer_Car_Price_2025.jpg",use_container_width=True)

# --- Title ---
st.title("🚗 Car Price Prediction")

# Sidebar Inputs
st.sidebar.header("Enter Details")

# Sliders for faster input
year = st.sidebar.slider("Year", min_value=1985, max_value=2025, value=2015, step=1)
km_driven = st.sidebar.slider("Kilometers Driven", min_value=0, max_value=300000, value=50000, step=1000)

fuel = st.sidebar.selectbox("Fuel Type", ("Petrol", "Diesel", "CNG", "LPG", "Electric"))
transmission = st.sidebar.selectbox("Transmission", ("Manual", "Automatic"))
owner = st.sidebar.selectbox("Owner", ("First Owner", "Second Owner", "Third Owner", "Fourth & Above"))
seller_type = st.sidebar.selectbox("Seller Type", ("Individual", "Dealer", "Trustmark Dealer"))

# Input DataFrame
input_df = pd.DataFrame({
    "year": [year],
    "km_driven": [km_driven],
    "fuel": [fuel],
    "transmission": [transmission],
    "owner": [owner],
    "seller_type": [seller_type],
})

st.subheader("📊 Your Input")
st.write(input_df)
st.markdown("---")

# --- Model Load (Backend only, no UI section) ---
model = None
default_model_path = "car_price_best_model.pkl"

if os.path.exists(default_model_path):
    try:
        with open(default_model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
else:
    st.error("❌ Model file not found. Please make sure car_price_best_model.pkl is in the same folder.")

# --- Prediction Button ---
if st.button("Predict Price", type="primary", use_container_width=True):
    if model is None:
        st.error("⚠️ No model available. Please train and save car_price_best_model.pkl")
    else:
        try:
            y_pred = model.predict(input_df)[0]
            st.success(f"💰 Predicted Price: ₹ {y_pred:,.0f}")
        except Exception as e:
            st.error(f"Prediction failed ❌\n\n{e}")
