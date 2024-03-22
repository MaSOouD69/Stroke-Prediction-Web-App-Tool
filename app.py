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
hypertension = st.selectbox('Do you have Hypertension?', ['No', 'Yes'])
heart_disease = st.selectbox('Do you have Heart Disease?', ['No', 'Yes'])
ever_married = st.selectbox('Have you ever been Married?', ['No', 'Yes'])
work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'Children', 'Never worked'])
residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0, value=90.0)
height = st.number_input('Height (in cm)', min_value=0.0)
weight = st.number_input('Weight (in kg)', min_value=0.0)
smoking_status = st.selectbox('Smoking Status', ['Never smoked', 'Sometimes', 'Formerly smoked', 'Smokes'])

# Calculate BMI
bmi = weight / ((height / 100) ** 2) if height > 0 and weight > 0 else 0

# Encode the inputs
features = {
    'gender': {'Male': 1, 'Female': 0, 'Other': 2}[gender],
    'age': age,
    'hypertension': {'Yes': 1, 'No': 0}[hypertension],
    'heart_disease': {'Yes': 1, 'No': 0}[heart_disease],
    'ever_married': {'Yes': 1, 'No': 0}[ever_married],
    'work_type': {'Private': 1, 'Self-employed': 2, 'Govt_job': 3, 'Children': 4, 'Never worked': 5}[work_type],
    'Residence_type': {'Urban': 1, 'Rural': 0}[residence_type],
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi,
    'smoking_status': {'Never smoked': 4, 'Sometimes': 3, 'Formerly smoked': 2, 'Smokes': 1}[smoking_status],
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

    # Display the calculated BMI and prediction result
    st.markdown(f"#### Your Calculated BMI: {bmi:.2f}")
    if prediction == 0:
        st.success('🎉 **Congratulations! You are not likely at risk of stroke.** Nonetheless, maintaining a healthy lifestyle is key. Always consult a healthcare professional for personalized advice.')
    else:
        st.error('⚠️ **Alert! You may be at risk of stroke.** It’s crucial to consult a healthcare professional for a comprehensive evaluation and advice.')
