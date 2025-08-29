import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

# Load car dataset
car = pd.read_csv('Cleaned_Car_data.csv')

# Streamlit App
st.title("🚗 Car Price Prediction App")

# Dropdowns and input fields
companies = sorted(car['company'].unique())
car_models = sorted(car['name'].unique())
years = sorted(car['year'].unique(), reverse=True)
fuel_types = car['fuel_type'].unique()

company = st.selectbox("Select Company", companies)
car_model = st.selectbox("Select Car Model", car_models)
year = st.selectbox("Select Year", years)
fuel_type = st.selectbox("Select Fuel Type", fuel_types)
driven = st.number_input("Enter Kilometers Driven", min_value=0, step=1000)

# Prediction button
if st.button("Predict Price"):
    input_data = pd.DataFrame(
        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
        data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)
    )
    
    prediction = model.predict(input_data)
    st.success(f"💰 Estimated Car Price: ₹ {np.round(prediction[0], 2)}")
