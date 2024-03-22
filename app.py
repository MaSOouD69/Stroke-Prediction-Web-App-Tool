import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load your trained models and scaler
model_decision_tree = joblib.load('model_filename.pkl')
model_logistic_regression = joblib.load('model_logistic.pkl')
scaler = joblib.load('scaler_filename.joblib')

# Function to convert height and weight from imperial to metric units
def convert_to_metric(height_ft, height_in, weight_lb):
    height_cm = (height_ft * 12 + height_in) * 2.54
    weight_kg = weight_lb * 0.453592
    return height_cm, weight_kg

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
work_type = st.selectbox('Employment Status', ['Private sector', 'Self-employed', 'Government employee', 'Unemployed', 'Student'])
residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0, value=90.0)
smoking_status = st.selectbox('Smoking Status', ['Never smoked', 'Sometimes', 'Formerly smoked', 'Smokes'])

# Option to choose metric or imperial units for height and weight
unit_system = st.selectbox('Choose unit system for Height and Weight', ['Metric', 'Imperial'])
if unit_system == 'Metric':
    height = st.number_input('Height (in cm)', min_value=0, step=1)
    weight = st.number_input('Weight (in kg)', min_value=0, step=1)
else:  # If imperial units selected, provide separate fields for feet and inches for height, and pounds for weight
    height_ft = st.number_input('Height (Feet)', min_value=0, step=1)
    height_in = st.number_input('Height (Inches)', min_value=0, max_value=11, step=1)
    weight_lb = st.number_input('Weight (in pounds)', min_value=0, step=1)
    height, weight = convert_to_metric(height_ft, height_in, weight_lb)

# Calculate BMI
bmi = weight / ((height / 100) ** 2) if height > 0 and weight > 0 else 0

# Encode the inputs
features = {
    'gender': {'Male': 1, 'Female': 0, 'Other': 2}[gender],
    'age': age,
    'hypertension': {'Yes': 1, 'No': 0}[hypertension],
    'heart_disease': {'Yes': 1, 'No': 0}[heart_disease],
    'ever_married': {'Yes': 1, 'No': 0}[ever_married],
    'work_type': {'Private sector': 1, 'Self-employed': 2, 'Government employee': 3, 'Unemployed': 4, 'Student': 5}[work_type],
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
