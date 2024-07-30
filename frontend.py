import streamlit as st
import joblib
import numpy as np

# Paths to the models and scaler
MODEL_PATH = 'F:/Water-Index-Prediction-main/Models/XGBoost_Model.pkl'
SCALER_PATH = 'F:/Water-Index-Prediction-main/Models/min_max_scaler.pkl'
MAPPING_DICT_PATH = 'F:/Water-Index-Prediction-main/Models/mapping_dict.pkl'

# Load the models and mappings
xgb_reg = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
mapping_dict = joblib.load(MAPPING_DICT_PATH)

# Streamlit app
st.title('Water Index Prediction')
st.sidebar.header('About the Project')
st.sidebar.write("""
    This project uses a machine learning model to predict the Water Index for different states based on various features.
    The features include state area, population, total water storage, groundwater level, total water demand, and rainfall.
    The model used is an XGBoost regressor, which is trained to provide accurate predictions for water management and planning.
""")

st.sidebar.subheader('Input State Details')

# User input
state = st.selectbox('State', list(mapping_dict.keys()))
state_area = st.number_input('State area (Km²)', min_value=0.0, step=0.1)
year = st.number_input('Year', min_value=1900, max_value=2100, step=1)
population = st.number_input('State Population', min_value=0, step=1)
water_storage = st.number_input('Total Water Storage in Reservoirs (mcm)', min_value=0.0, step=0.1)
groundwater_level = st.number_input('Groundwater Level (mbgl)', min_value=0.0, step=0.1)
water_demand = st.number_input('Total Water Demand (BCM)', min_value=0.0, step=0.1)
rain_water = st.number_input('Rain Water (mm)', min_value=0.0, step=0.1)

# Prepare the input data
data_to_predict = [[
    mapping_dict[state],  # Encoded state feature
    state_area,
    year,
    population,
    water_storage,
    groundwater_level,
    water_demand,
    rain_water
]]

# Ensure the correct number of features
if len(data_to_predict[0]) != scaler.n_features_in_:
    st.error(f"Error: Expected {scaler.n_features_in_} features, but got {len(data_to_predict[0])}.")
else:
    # Scale the data
    scaled_data = scaler.transform(data_to_predict)
    
    # Make prediction
    prediction = xgb_reg.predict(scaled_data)
    
    # Output formatting
    st.subheader('Prediction Result')
    st.write(f'**Predicted Water Index for {state}:**')
    st.write(f'```{prediction[0]:.4f}```')

    # Display additional information
    st.write("""
        **Note:** The Water Index is a measure used to assess water availability and stress. Higher values indicate poor water availability.
    """)
