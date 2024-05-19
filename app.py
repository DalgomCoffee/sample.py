import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model = load_model('finaltrain.h5')

# Function to make predictions
def predict(input_data):
    return model.predict(input_data)

# Streamlit app
st.title("My Keras Model Prediction App")

# Input fields for user data
st.header("Enter input data for prediction")

# Assuming your model takes three numerical inputs
input1 = st.number_input("Input 1", value=0.0)
input2 = st.number_input("Input 2", value=0.0)
input3 = st.number_input("Input 3", value=0.0)

# Make a numpy array from the inputs
input_data = np.array([[input1, input2, input3]])

# When the user clicks the 'Predict' button
if st.button("Predict"):
    prediction = predict(input_data)
    st.write(f"The predicted value is: {prediction[0][0]}")
