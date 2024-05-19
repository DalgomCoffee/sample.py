import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, UnidentifiedImageError

# Load the pre-trained model
model = load_model('finaltrain.h5')

# Define the class names for weather conditions
class_names = ['Shine', 'Rain', 'Sunrise', 'Cloudy']

# Preprocess the image for prediction
def preprocess_image(image, target_size=(60, 40)):
    try:
        # Resize and convert to grayscale
        image = image.resize(target_size).convert('L')
        # Convert to numpy array and normalize pixel values
        image = np.array(image) / 255.0
        # Flatten and add batch dimension
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        st.error(f"An error occurred during image preprocessing: {str(e)}")
        return None

# Make predictions using the pre-trained model
def predict(image):
    try:
        p_image = preprocess_image(image)
        if p_image is not None:
            return model.predict(p_image)
        else:
            return None
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        return None

# Streamlit app layout
st.title("FINAL EXAM: WEATHER PREDICTION")
uploaded_file = st.file_uploader("Upload a weather image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Read the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Make prediction
        prediction = predict(image)
        if prediction is not None:
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            predicted_class = class_names[predicted_class_index]
            st.success(f"Prediction: {predicted_class}")
    except UnidentifiedImageError:
        st.error("The uploaded file could not be identified as an image. Please upload a valid image file.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Instructions
st.markdown("""
### Instructions:
1. Upload the weather image (jpg, png, jpeg).
2. Prediction will be displayed.
""")
