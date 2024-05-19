import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from PIL import Image

# Load the trained model
model = load_model('finaltrain.h5')

# Function to preprocess the input image
def preprocess_image(image, target_size=(64, 64)):
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make predictions
def predict(image):
    processed_image = preprocess_image(image)
    return model.predict(processed_image)

# Streamlit app
st.title("Weather Prediction Model Accuracy Checker")

# Upload image
uploaded_file = st.file_uploader("Upload a weather image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make prediction
    prediction = predict(image)
    predicted_temperature = prediction[0][0]  # Assuming the model outputs temperature

    st.write(f"Predicted Temperature: {predicted_temperature:.2f}")

    # Option to enter actual temperature
    actual_temperature = st.number_input("Enter Actual Temperature", format="%.2f")

    if st.button("Calculate Accuracy"):
        # Calculate accuracy metrics
        mse_temp = mean_squared_error([actual_temperature], [predicted_temperature])
        mae_temp = mean_absolute_error([actual_temperature], [predicted_temperature])

        st.write(f"Mean Squared Error (Temperature): {mse_temp:.2f}")
        st.write(f"Mean Absolute Error (Temperature): {mae_temp:.2f}")

# Instructions for the user
st.markdown("""
### Instructions:
1. Upload a weather-related image (jpg, png, jpeg).
2. Enter the actual temperature corresponding to the image.
3. Click "Calculate Accuracy" to see how well the model predicts the temperature.
""")
