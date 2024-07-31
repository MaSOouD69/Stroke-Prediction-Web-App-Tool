import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load your trained models and scaler
model_decision_tree = joblib.load('model_filename.pkl')
model_logistic_regression = joblib.load('model_logistic.pkl')
scaler = joblib.load('scaler_filename.joblib')

# Streamlit application for user input
st.title('Stroke Risk Prediction Tool')

st.markdown("""
This tool predicts the likelihood of having a stroke based on personal health information.
Please fill in the details below and click "Predict" to see your risk assessment.
""")

# User inputs
gender = st.selectbox('Gender', ['Male', 'Female', 'Other'], help="Select your gender.")
age = st.number_input('Age', min_value=0, help="Enter your age. This field is required.")
hypertension = st.selectbox('Hypertension', ['No', 'Yes'], help="Do you have hypertension?")
heart_disease = st.selectbox('Heart Disease', ['No', 'Yes'], help="Do you have heart disease?")
ever_married = st.selectbox('Ever Married', ['No', 'Yes'], help="Have you ever been married?")
work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Government Job', 'Children', 'Never Worked'], help="Select your type of employment.")
residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'], help="What type of area do you live in?")
avg_glucose_level = st.number_input('Average Glucose Level (mg/dL)', min_value=0, value=90, help="Enter your average glucose level.")
smoking_status = st.selectbox('Smoking Status', ['Never smoked', 'Sometimes', 'Formerly smoked', 'Smokes'])

unit_system = st.selectbox('Choose Unit System', ['Metric (cm, kg)', 'Imperial (inches, pounds)'])

if unit_system == 'Metric (cm, kg)':
    height = st.number_input('Height (cm)', min_value=0, step=1, help="Enter your height in centimeters.")
    weight = st.number_input('Weight (kg)', min_value=0, step=1, help="Enter your weight in kilograms.")
elif unit_system == 'Imperial (inches, pounds)':
    height = st.number_input('Height (inches)', min_value=0, step=1, help="Enter your height in inches.")
    weight = st.number_input('Weight (pounds)', min_value=0, step=1, help="Enter your weight in pounds.")

# Convert height and weight to cm and kg if entered in imperial units
if unit_system == 'Imperial (inches, pounds)':
    height = height * 2.54  # Convert inches to cm
    weight = weight * 0.453592  # Convert pounds to kg

# Calculate BMI
bmi = weight / ((height / 100) ** 2) if height > 0 and weight > 0 else 0

# Encode the inputs
features = {
    'gender': {'Male': 1, 'Female': 0, 'Other': 2}[gender],
    'age': age,
    'hypertension': {'Yes': 1, 'No': 0}[hypertension],
    'heart_disease': {'Yes': 1, 'No': 0}[heart_disease],
    'ever_married': {'Yes': 1, 'No': 0}[ever_married],
    'work_type': {'Private': 1, 'Self-employed': 2, 'Government Job': 3, 'Children': 4, 'Never Worked': 5}[work_type],
    'Residence_type': {'Urban': 1, 'Rural': 0}[residence_type],
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi,
    'smoking_status': {'Never smoked': 4, 'Sometimes': 3, 'Formerly smoked': 2, 'Smokes': 1}[smoking_status],
    'missing_bmi': 0
}

# Convert features to DataFrame
input_df = pd.DataFrame([features])

# Scale the input data
scaled_input = scaler.transform(input_df)

# Model prediction
model_option = st.selectbox('Select Model', ['Decision Tree', 'Logistic Regression'])
if st.button('Predict'):
    if age > 0 and height > 0 and weight > 0:
        prediction = model_decision_tree.predict(scaled_input) if model_option == 'Decision Tree' else model_logistic_regression.predict(scaled_input)

        # Display results
        st.markdown(f"#### Your Calculated BMI: {bmi:.2f}")
        if prediction == 0:
            st.success('🎉 **Congratulations! You are not likely at risk of stroke.** Nonetheless, maintaining a healthy lifestyle is key.')
        else:
            st.error('⚠️ **Alert! You may be at risk of stroke.** Consult a healthcare professional for a comprehensive evaluation.')
    else:
        st.warning('Please ensure all fields are correctly filled and try again.')

# Analysis on BMI
if bmi > 25:
    st.warning('⚠️ **You may be overweight.** Consider maintaining a healthy weight.')
elif bmi > 0:
    st.info('💡 **Your BMI is within the healthy range.**')
