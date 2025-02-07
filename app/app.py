import streamlit as st
import pandas as pd
import joblib

# Title of the app
st.title('Fire Weather Index Prediction App')

# Sidebar inputs for user data
FFMC = st.sidebar.slider('Fine Fuel Moisture Code (FFMC)', 0.0, 100.0, 86.2)
DMC = st.sidebar.slider('Duff Moisture Code (DMC)', 0.0, 150.0, 26.2)
DC = st.sidebar.slider('Drought Code (DC)', 0.0, 800.0, 94.3)
ISI = st.sidebar.slider('Initial Spread Index (ISI)', 0.0, 50.0, 5.1)
Temperature = st.sidebar.slider('Temperature (°C)', -10.0, 50.0, 8.2)
RH = st.sidebar.slider('Relative Humidity (%)', 0, 100, 51)
Rain = st.sidebar.slider('Rain (mm)', 0.0, 10.0, 0.0)
Ws = st.sidebar.slider('Wind Speed (km/h)', 0.0, 30.0, 2.2)
Region = st.sidebar.selectbox('Region', [0, 1])  # Example: 0 for one region, 1 for another
BUI = st.sidebar.slider('Buildup Index (BUI)', 0.0, 100.0, 7.9)
Classes = st.sidebar.selectbox('Fire Class (0 = Low, 1 = High)', [0, 1])

# Create input dictionary
input_data = {
    'FFMC': FFMC,
    'DMC': DMC,
    'DC': DC,
    'ISI': ISI,
    'Temperature': Temperature,
    'RH': RH,
    'Rain': Rain,
    'Ws': Ws,
    'Region': Region,
    'BUI': BUI,
    'Classes': Classes
}

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# Load the trained model
model = joblib.load('../models/model_pipeline.pkl')

# Make prediction
if st.sidebar.button('Predict FWI'):
    prediction = model.predict(input_df)
    st.write(f'### Predicted Fire Weather Index (FWI): {prediction[0]:.2f}')
