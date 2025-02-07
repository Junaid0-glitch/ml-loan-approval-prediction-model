import numpy as np
import streamlit as st
import joblib

# Load the scaler and model
scaler = joblib.load('scaler.joblib')
model = joblib.load('forest_model.joblib')

# Streamlit UI
st.title("Loan Approval Prediction")
st.write("Enter the loan details to check if your loan will be Approved or Rejected.")

# Collect user inputs
no_of_dependents = st.number_input('Number of Dependents', min_value=0, max_value=10, step=1)
self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
income_annum = st.number_input('Annual Income (in INR)', min_value=0)
loan_amount = st.number_input('Loan Amount (in INR)', min_value=0)
loan_term = st.slider('Loan Term (in months)', min_value=1, max_value=360)
cibil_score = st.number_input('CIBIL Score', min_value=300, max_value=900)
residential_assets_value = st.number_input('Residential Assets Value (in INR)', min_value=0)
commercial_assets_value = st.number_input('Commercial Assets Value (in INR)', min_value=0)
luxury_assets_value = st.number_input('Luxury Assets Value (in INR)', min_value=0)
bank_asset_value = st.number_input('Bank Asset Value (in INR)', min_value=0)
education_not_graduate = st.selectbox('Education', ['Graduate', 'Not Graduate'])

# Convert categorical features to numerical
self_employed = 1 if self_employed == 'Yes' else 0
education_not_graduate = 1 if education_not_graduate == 'Not Graduate' else 0

# Prepare the numerical features for scaling (excluding categorical ones)
numerical_features = np.array([[no_of_dependents, income_annum, loan_amount, loan_term, cibil_score,
                                residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value]])

# Scale the numerical features
scaled_numerical_features = scaler.transform(numerical_features)

# Combine the scaled numerical features with the categorical ones
final_features = np.hstack((scaled_numerical_features, [[self_employed, education_not_graduate]]))

# Button to make prediction
if st.button("Predict"):
    prediction = model.predict(final_features)
    if prediction == 1:
        st.success("Loan Approved!")
    else:
        st.error("Loan Rejected.")

