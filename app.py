
import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load your trained model (Python reads the binary file here!)
model = joblib.load('loan_model.pkl')

# 2. Build the UI Titles
st.title("🏦 Bank Loan Prediction AI")
st.write("Enter the applicant's details below to see if their loan will be approved.")

# 3. Create input fields for the user
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

with col2:
    applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=150)
    loan_amount_term = st.number_input("Loan Term (Days)", min_value=0, value=360)
    credit_history = st.selectbox("Credit History", ["Good (1.0)", "Bad (0.0)"])

# 4. Map the text inputs to the exact 1s and 0s your model expects
gender_val = 1 if gender == "Male" else 0
married_val = 1 if married == "Yes" else 0
education_val = 1 if education == "Graduate" else 0
self_employed_val = 1 if self_employed == "Yes" else 0

if dependents == "3+":
    dep_val = 4
else:
    dep_val = int(dependents)

if property_area == "Rural":
    prop_val = 0
elif property_area == "Semiurban":
    prop_val = 1
else:
    prop_val = 2

credit_val = 1.0 if credit_history == "Good (1.0)" else 0.0

# 5. When the user clicks the Predict button
if st.button("Predict Loan Status"):
    # Package the data exactly how the model saw it during training
    input_data = pd.DataFrame([[
        gender_val, married_val, dep_val, education_val, self_employed_val,
        applicant_income, coapplicant_income, loan_amount, loan_amount_term,
        credit_val, prop_val
    ]], columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
                 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                 'Loan_Amount_Term', 'Credit_History', 'Property_Area'])
    
    # Make the prediction
    prediction = model.predict(input_data)
    
    # Display the result to the screen
    st.divider()
    if prediction[0] == 1:
        st.success("🎉 LOAN APPROVED!")
        st.balloons()
    else:
        st.error("❌ LOAN REJECTED.")