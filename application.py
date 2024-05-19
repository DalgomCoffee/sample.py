import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, UnidentifiedImageError

# Load the model
model = load_model('finaltrain.h5')
class_names = ['Rain', 'Shine', 'Cloudy', 'Sunrise']

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess the image to the required input shape of the model.

    Args:
        image (PIL.Image): The input image to preprocess.
        target_size (tuple): The target size to resize the image.

    Returns:
        np.ndarray: The preprocessed image ready for prediction.
    """
    image = image.resize(target_size)
    image = image.convert('RGB')  # Ensure image has 3 channels (RGB)
    image = np.array(image)
    image = image / 255.0  # Normalize the image to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict_weather(image):
    """
    Predict the weather condition from the preprocessed image.

    Args:
        image (PIL.Image): The input image to predict.

    Returns:
        str: The predicted weather condition.
    """
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    
    if predicted_class_index < len(class_names):
        return class_names[predicted_class_index]
    else:
        return "Unknown"

# Streamlit app
st.title("Weather Prediction App")
uploaded_file = st.file_uploader("Upload a weather image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        weather_prediction = predict_weather(image)
        st.success(f"Prediction: {weather_prediction}")
    except UnidentifiedImageError:
        st.error("Please upload a valid image.")
    except Exception as e:
        st.error("An error occurred during prediction.")
        st.error(str(e))

st.markdown("""
### Instructions:
1. Upload a weather image (jpg, png, jpeg).
2. The prediction will be displayed.
""")
