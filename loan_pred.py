import streamlit as st
import numpy as np
import joblib
st.title("Loan Approval Prediction")


# Collect input data from user
Currency = st.selectbox(["USD","INR"])
ApplicantIncome = st.number_input("Applicant Income")
CoapplicantIncome = st.number_input("Coapplicant Income")
LoanAmount = st.number_input("Loan Amount")
Loan_Amount_Term = st.selectbox("Loan Term (in days)", [3600, 1800, 1200, 600, 360, 180, 120, 60])
Credit_History = st.selectbox("Credit History (1 = Good, 0 = Bad)", [1, 0])
property_area = st.selectbox("Property Area", ["Urban", "Semi-Urban", "Rural"])

# Map property area to encoded values
property_area_map = {"Urban": 2, "Semi-Urban": 1, "Rural": 0}
Property_Area = property_area_map[property_area]
if Currency == "INR":
    ApplicantIncome /= 80
    CoapplicantIncome /= 80
    LoanAmount /= 80
    
# Prepare input data for the model
features = np.array([[ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area]])

# Load the trained model
model = joblib.load("models/loan_approval_01.pkl")

# Make predictions when button is clicked
if st.button("Predict Loan Approval"):
    if LoanAmount == 0:
        st.error("Loan amount can't be zero...")
    else: 
        prediction = model.predict(features)
        if prediction[0] == 1:
            st.success("Loan Approved ✅")
        else:
            st.error("Loan Not Approved ❌")


