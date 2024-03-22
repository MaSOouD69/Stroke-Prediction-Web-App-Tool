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
gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
age = st.number_input('Age', min_value=0)
hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
heart_disease = st.selectbox('Heart Disease', ['No', 'Yes'])
ever_married = st.selectbox('Ever Married', ['No', 'Yes'])
work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Government Job', 'Children', 'Never Worked'])
residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
avg_glucose_level = st.number_input('Average Glucose Level (mg/dL)', min_value=0)
smoking_status = st.selectbox('Smoking Status', ['never smoked', 'Sometimes', 'formerly smoked', 'smokes'])

unit_system = st.selectbox('Choose Unit System', ['Metric (cm, kg)', 'Imperial (inches, pounds)'])

if unit_system == 'Metric (cm, kg)':
    height = st.number_input('Height (cm)', min_value=0, step=1)
    weight = st.number_input('Weight (kg)', min_value=0, step=1)
elif unit_system == 'Imperial (inches, pounds)':
    height = st.number_input('Height (inches)', min_value=0, step=1)
    weight = st.number_input('Weight (pounds)', min_value=0, step=1)

if not any([hypertension, heart_disease, ever_married]):
    avg_glucose_level = st.number_input('Average Glucose Level (mg/dL)', min_value=0, value=90)

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
    'smoking_status': {'never smoked': 4, 'Sometimes': 3, 'formerly smoked': 2, 'smokes': 1}[smoking_status],
    'missing_bmi': 0  # Including for model compatibility, but not displaying to the user
}

# Convert features to DataFrame
input_df = pd.DataFrame([features])

# Scale the input data
scaled_input = scaler.transform(input_df)

# Model prediction
model_option = st.selectbox('Select Model', ['Decision Tree', 'Logistic Regression'])
if st.button('Predict'):
    prediction = model_decision_tree.predict(scaled_input) if model_option == 'Decision Tree' else model_logistic_regression.predict(scaled_input)

    # Creative display of the calculated BMI and prediction result
    st.markdown(f"#### Your Calculated BMI: {bmi:.2f}")
    if prediction == 0:
        st.success('🎉 **Congratulations! You are not likely at risk of stroke.** Nonetheless, maintaining a healthy lifestyle is key. Always consult a healthcare professional for personalized advice.')
    else:
        st.error('⚠️ **Alert! You may be at risk of stroke.** It’s crucial to consult a healthcare professional for a comprehensive evaluation and advice.')

    # Analysis on BMI
    if bmi > 25:
        st.warning('⚠️ **You may be overweight.** It is recommended to maintain a healthy weight to reduce the risk of stroke.')
    else:
        st.info('💡 **Your BMI is within the healthy range.** However, maintaining a healthy lifestyle is always beneficial.')
