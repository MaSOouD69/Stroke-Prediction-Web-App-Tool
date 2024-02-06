{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "40f5df65",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import joblib\n",
    "\n",
    "# Load the saved model\n",
    "model = joblib.load('model_filename.pkl')\n",
    "\n",
    "\n",
    "# Create a title for your app\n",
    "st.title('Machine Learning App')\n",
    "\n",
    "# Add a text input for user input\n",
    "user_input = st.text_input('Enter text:', '')\n",
    "\n",
    "# Add a button to trigger predictions\n",
    "if st.button('Predict'):\n",
    "    # Use the loaded model to make predictions\n",
    "    prediction = model.predict([user_input])  # You may need to preprocess the input\n",
    "    st.write('Prediction:', prediction)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
