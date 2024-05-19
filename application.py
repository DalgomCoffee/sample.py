import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, UnidentifiedImageError

model = load_model('finaltrain.h5')
class_names = ['Rain', 'Shine', 'Cloudy', 'Sunrise']

def preprocess_image(image, target_size=(60, 40)):
    try:
        # Resize and convert to grayscale
        image = image.resize(target_size).convert('L')
        # Normalize pixel values
        image = np.array(image) / 255.0
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        return image
    
def predict(image):
    p_image = preprocess_image(image)
    return model.predict(p_image)

st.title("FINAL EXAM: WEATHER PREDICTION")
uploaded_file = st.file_uploader("Upload a weather image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        prediction = predict(image)
        
        prediction=predict(image)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class = class_names[predicted_class_index]
        st.success(f"Prediction: {predicted_class}")
    except UnidentifiedImageError:
        st.error("Pls put valid image.")
 
st.markdown("""
### Instructions:
1. Upload the weather image (jpg, png, jpeg).

2.Prediction will display
""")


