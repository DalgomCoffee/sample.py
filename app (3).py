import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, UnidentifiedImageError

# Load the trained model
model = load_model('finaltrain.h5')
class_names = ['Rain', 'Shine', 'Cloudy', 'Sunrise']

# Function to preprocess the input image
def preprocess_image(image, target_size=(40, 60)):
    image = image.resize(target_size)
    image = image.convert('L')  # Convert to grayscale
    image = np.array(image)
    image = image / 255.0  # Normalize the image
    image = image.flatten()  # Flatten the image to a 1D array of size 2400
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make predictions
def predict(image):
    p_image = preprocess_image(image)
    return model.predict(p_image)

# Streamlit app
st.title("Final Exam: Model Deployment in the Cloud")

# Upload image
uploaded_file = st.file_uploader("Upload a weather image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Read the image file
        image = Image.open(uploaded_file)
        
        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make prediction
        prediction = predict(image)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class = class_names[predicted_class_index]

        # Display the prediction
        st.success(f"Prediction: {predicted_class}")

    except UnidentifiedImageError:
        st.error("The uploaded file could not be identified as an image. Please upload a valid image file.")
 

# Instructions for the user
st.markdown("""
### Instructions:
1. Upload the weather image (jpg, png, jpeg).
2. Wait for the prediction to be displayed.
""")
