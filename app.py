import streamlit as st
import pickle
import numpy as np

# Load trained model
# Make sure model.pkl is in the same folder as app.py
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🍷 Wine Quality Prediction App")

st.write("Enter wine chemical properties to predict quality")

# Input fields
fixed_acidity = st.number_input("Fixed Acidity", value=7.9)
volatile_acidity = st.number_input("Volatile Acidity", value=0.35)
citric_acid = st.number_input("Citric Acid", value=0.46)
residual_sugar = st.number_input("Residual Sugar", value=3.6)
chlorides = st.number_input("Chlorides", value=0.078)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=15)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=37)
density = st.number_input("Density", value=0.9973)
pH = st.number_input("pH", value=3.35)
sulphates = st.number_input("Sulphates", value=0.86)
alcohol = st.number_input("Alcohol", value=12.8)

# Prediction button
if st.button("Predict Wine Quality"):
    sample_data = np.array([[fixed_acidity,
                              volatile_acidity,
                              citric_acid,
                              residual_sugar,
                              chlorides,
                              free_sulfur_dioxide,
                              total_sulfur_dioxide,
                              density,
                              pH,
                              sulphates,
                              alcohol]])

    prediction = model.predict(sample_data)

    st.success(f"🍾 Predicted Wine Quality: {prediction[0]}")
