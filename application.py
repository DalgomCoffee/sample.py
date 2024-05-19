import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import numpy as np

# Load the pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
class_names = ['Rainy', 'Shine', 'Cloudy', 'Sunrise']

# Preprocess the image for MobileNetV2
def preprocess_image(image):
    img = image.resize((224, 224))
    img = img.convert('RGB')
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Predict the weather condition
def predict_weather(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction)
    if predicted_class_index < len(class_names):
        predicted_class = class_names[predicted_class_index]
        return predicted_class
    else:
        print("Predicted class index:", predicted_class_index)
        print("Length of class_names list:", len(class_names))
        return "Unknown"

# Streamlit app
st.title("Weather Prediction")
uploaded_file = st.file_uploader("Upload a weather image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        weather_prediction = predict_weather(image)
        st.success(f"Predicted Weather Condition: {weather_prediction}")
    except UnidentifiedImageError:
        st.error("Please upload a valid image.")

st.markdown("""
### Instructions:
1. Upload the weather image (jpg, png, jpeg).
2. The predicted weather condition will be displayed as Rainy, Shine, Cloudy, or Sunrise.
""")
