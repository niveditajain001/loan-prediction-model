import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Loan AI", layout="wide")

# --- 2. DATA LOADING & MODEL TRAINING ---
# @st.cache_resource ensures the model only trains once when the app starts
@st.cache_resource
def train_model():
    # Load your exact dataset
    df = pd.read_csv('dataset/train_u6lujuX_CVtuZ9i.csv')
    
    # Preprocessing (Exactly as you wrote it)
    df.columns = df.columns.str.strip()
    df.dropna(inplace=True)
    df.replace({
        'Loan_Status': {'Y': 1, 'N': 0},
        'Married': {'No': 0, 'Yes': 1},
        'Gender': {'Male': 1, 'Female': 0},
        'Self_Employed': {'Yes': 1, 'No': 0},
        'Education': {'Graduate': 1, 'Not Graduate': 0, 'Non Graduate': 0},
        'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2}
    }, inplace=True)
    
    df = df.replace(to_replace='3+', value=4)
    df['Dependents'] = pd.to_numeric(df['Dependents'])
    
    # Split data
    X = df.drop(columns=['Loan_ID', 'Loan_Status'])
    Y = df['Loan_Status'].astype('int')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)
    
    # Scale numerical data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM Model
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train_scaled, Y_train)
    
    return classifier, scaler, X_test_scaled, Y_test

# Load the trained components
model, scaler, X_test_scaled, Y_test = train_model()

# --- 3. SIDEBAR NAVIGATION ---
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Select a Page:", ["1. Live Prediction Engine", "2. Model Evaluation Data"])

# --- 4. PAGE 1: PREDICTION ENGINE ---
if page == "1. Live Prediction Engine":
    st.title("💳 Loan Risk Evaluation Engine")
    
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", [0, 1, 2, 4])
        education = st.selectbox("Education", ["Graduate", "Non Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        applicant_income = st.number_input("Applicant Income", value=5000)
        
    with col2:
        coapplicant_income = st.number_input("Coapplicant Income", value=0)
        loan_amount = st.number_input("Loan Amount", value=150)
        loan_term = st.number_input("Loan Amount Term (Days)", value=360)
        credit_history = st.selectbox("Credit History (1=Good, 0=Bad)", [1.0, 0.0])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    if st.button("Predict Loan Status", type="primary"):
        # Map user inputs to numerical values matching training data
        input_df = pd.DataFrame([{
            'Gender': 1 if gender == "Male" else 0,
            'Married': 1 if married == "Yes" else 0,
            'Dependents': dependents,
            'Education': 1 if education == "Graduate" else 0,
            'Self_Employed': 1 if self_employed == "Yes" else 0,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_term,
            'Credit_History': credit_history,
            'Property_Area': 2 if property_area == "Urban" else (1 if property_area == "Semiurban" else 0)
        }])
        
        # Scale the inputs and predict
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        
        st.divider()
        if prediction[0] == 1:
            st.success("✅ Verdict: Loan Approved")
            st.balloons()
        else:
            st.error("❌ Verdict: Loan Denied")

# --- 5. PAGE 2: EVALUATION METRICS ---
elif page == "2. Model Evaluation Data":
    st.title("📊 Model Performance Metrics")
    
    # Calculate real metrics from the test split
    test_predictions = model.predict(X_test_scaled)
    acc = accuracy_score(Y_test, test_predictions)
    conf_matrix = confusion_matrix(Y_test, test_predictions)
    class_report = classification_report(Y_test, test_predictions)
    
    st.write(f"**Model Accuracy:** {acc * 100:.2f}%")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Confusion Matrix:**")
        st.dataframe(conf_matrix)
    with col2:
        st.write("**Classification Report:**")
        st.text(class_report)
        
    st.divider()
    
    st.subheader("Algorithm Comparison")
    # Display comparison table
    comp_df = pd.DataFrame({
        "Algorithm": ["Support Vector Machine (Selected)", "Logistic Regression", "K-Nearest Neighbors"],
        "Handling of Data": ["Effective in high-dimensional spaces", "Fast but assumes linear relationships", "Requires extensive scaling, distance-based"],
        "Pros": ["Robust against overfitting, precise boundaries", "Highly interpretable, fast", "Intuitive logic, visual"]
    })
    st.dataframe(comp_df, use_container_width=True)
# import streamlit as st
# import pandas as pd
# import joblib
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# # 1. Load your trained model (Python reads the binary file here!)
# model = joblib.load('loan_model.pkl')

# # 2. Build the UI Titles
# st.title("🏦 Bank Loan Prediction AI")
# st.write("Enter the applicant's details below to see if their loan will be approved.")

# # 3. Create input fields for the user
# col1, col2 = st.columns(2)

# with col1:
#     gender = st.selectbox("Gender", ["Male", "Female"])
#     married = st.selectbox("Married", ["Yes", "No"])
#     dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
#     education = st.selectbox("Education", ["Graduate", "Not Graduate"])
#     self_employed = st.selectbox("Self Employed", ["Yes", "No"])
#     property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

# with col2:
#     applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000)
#     coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0, value=0)
#     loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=150)
#     loan_amount_term = st.number_input("Loan Term (Days)", min_value=0, value=360)
#     credit_history = st.selectbox("Credit History", ["Good (1.0)", "Bad (0.0)"])

# # 4. Map the text inputs to the exact 1s and 0s your model expects
# gender_val = 1 if gender == "Male" else 0
# married_val = 1 if married == "Yes" else 0
# education_val = 1 if education == "Graduate" else 0
# self_employed_val = 1 if self_employed == "Yes" else 0

# if dependents == "3+":
#     dep_val = 4
# else:
#     dep_val = int(dependents)

# if property_area == "Rural":
#     prop_val = 0
# elif property_area == "Semiurban":
#     prop_val = 1
# else:
#     prop_val = 2

# credit_val = 1.0 if credit_history == "Good (1.0)" else 0.0

# # 5. When the user clicks the Predict button
# if st.button("Predict Loan Status"):
#     # Package the data exactly how the model saw it during training
#     input_data = pd.DataFrame([[
#         gender_val, married_val, dep_val, education_val, self_employed_val,
#         applicant_income, coapplicant_income, loan_amount, loan_amount_term,
#         credit_val, prop_val
#     ]], columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
#                  'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
#                  'Loan_Amount_Term', 'Credit_History', 'Property_Area'])
    
#     # Make the prediction
#     prediction = model.predict(input_data)
    
#     # Display the result to the screen
#     st.divider()
#     if prediction[0] == 1:
#         st.success("🎉 LOAN APPROVED!")
#         st.balloons()
#     else:
#         st.error("❌ LOAN REJECTED.")