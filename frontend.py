import streamlit as st
import joblib
import pandas as pd

# Updated paths for the new model, scaler, and encoder
MODEL_PATH = 'F:/Water-Index-Prediction-main/Models/RandomForest (1).joblib'
SCALER_PATH = 'F:/Water-Index-Prediction-main/Models/scaler.joblib'
ENCODER_PATH = 'F:/Water-Index-Prediction-main/Models/label_encoder.joblib'

# Load the model, scaler, and encoder
rf_reg = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

st.title('Water Index Prediction')

st.write("""
    **Please enter the details below to get the predicted Water Index.** 
    The features include state, year, state population, and total water demand.
""")

st.sidebar.header('About the Project')
st.sidebar.write("""
    This project uses a machine learning model to predict the Water Index for different states based on various features.
    The features include state, year, state population, and total water demand.
    The model used is a Random Forest regressor, which is trained to provide accurate predictions for water management and planning.
""")

st.sidebar.subheader('Water Stress Legend')
st.sidebar.write("""
    - **0-3:** Low water stress
    - **4-6:** Moderate water stress
    - **6+:** High water stress
""")

state = st.selectbox('State', list(label_encoder.classes_))
year = st.number_input('Year', min_value=1900, max_value=2100, step=1)
state_population = st.number_input('State Population', min_value=0, step=1)
water_demand = st.number_input('Total Water Demand (BCM)', min_value=0.0, step=0.1)

# Encode state and prepare data for prediction
state_encoded = label_encoder.transform([state])[0]
data_to_predict = pd.DataFrame([[
    state_encoded,
    year,
    state_population,
    water_demand
]], columns=['State Name', 'Year', 'State Population', 'Total Water Demand(BCM)'])

# Check if the number of features matches
if data_to_predict.shape[1] != scaler.n_features_in_:
    st.error(f"Error: Expected {scaler.n_features_in_} features, but got {data_to_predict.shape[1]}.")
else:
    scaled_data = scaler.transform(data_to_predict)
    prediction = rf_reg.predict(scaled_data)
    
    st.subheader('Prediction Result')
    st.write(f'**Predicted Water Index for {state}:**')
    st.write(f'```{prediction[0]:.4f}```')

    st.write("""
        **Note:** The Water Index is a measure used to assess water availability and stress. Higher values indicate poor water availability.
    """)
