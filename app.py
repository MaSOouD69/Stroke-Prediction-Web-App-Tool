import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load your trained models and scaler
model_decision_tree = joblib.load('model_filename.pkl')
model_logistic_regression = joblib.load('model_logistic.pkl')
scaler = joblib.load('scaler_filename.joblib')

# Streamlit application for user input
st.title('Stroke Prediction Assistant')

st.markdown("""
    This application helps in predicting the likelihood of a stroke.
    Please input the required fields and press predict to see the results.
""")

# Create input fields for all features
gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
age = st.number_input('Age', min_value=0)
hypertension = st.selectbox('Hypertension', [0, 1])  # Assuming 0 = No, 1 = Yes
heart_disease = st.selectbox('Heart Disease', [0, 1])  # Assuming 0 = No, 1 = Yes
ever_married = st.selectbox('Ever Married', ['No', 'Yes'])
work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0)
bmi = st.number_input('BMI', min_value=0.0)
smoking_status = st.selectbox('Smoking Status', ['never smoked', 'Unknown', 'formerly smoked', 'smokes'])
missing_bmi = st.selectbox('Missing BMI', [0, 1])  # Assuming 0 = No, 1 = Yes

# Mapping categorical inputs to numerical format
gender = {'Male': 1, 'Female': 0, 'Other': 0}[gender]
residence_type = {'Urban': 1, 'Rural': 0}[residence_type]
work_type = {'Private': 1, 'Self-employed': 2, 'children': 3, 'Govt_job': 4, 'Never_worked': 3}[work_type]
ever_married = {'Yes': 1, 'No': 0}[ever_married]
smoking_status = {'never smoked': 3, 'Unknown': 4, 'formerly smoked': 1, 'smokes': 2}[smoking_status]

# Prepare the input data as a DataFrame
input_data = pd.DataFrame([[gender, age, hypertension, heart_disease, ever_married, 
                            work_type, residence_type, avg_glucose_level, bmi, 
                            smoking_status, missing_bmi]])

# Scale the input data using the loaded scaler
scaled_input_data = scaler.transform(input_data)

# Select model
model_option = st.selectbox('Select Model', ['Decision Tree', 'Logistic Regression'])

if st.button('Predict'):
    if model_option == 'Decision Tree':
        prediction = model_decision_tree.predict(scaled_input_data)
    elif model_option == 'Logistic Regression':
        prediction = model_logistic_regression.predict(scaled_input_data)

    # User-friendly message based on prediction
    if prediction == 0:
        st.success('You are not likely at risk of stroke. However, always consult a healthcare professional for health-related advice.')
    else:
        st.error('You are at risk of stroke. Please consult a healthcare professional immediately.')

