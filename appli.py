import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, UnidentifiedImageError

# Load the trained model and class names
model = load_model('finaltrain.h5')
class_names = ['Rain', 'Shine', 'Cloudy', 'Sunrise']

# Preprocess the image
def preprocess_image(image, target_size=(40, 60)):
    image = image.resize(target_size)
    image = image.convert('L')  # Convert to grayscale
    image = np.array(image)
    image = image / 255.0  # Normalize pixel values
    image = image.flatten()  # Flatten image array
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to predict the class of the uploaded image
def predict(image):
    p_image = preprocess_image(image)
    return model.predict(p_image)

# Streamlit app layout
st.title("Final Exam: Model Deployment in the Cloud")

# File uploader
uploaded_file = st.file_uploader("Upload a weather image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make prediction
        prediction = predict(image)
        top_classes_indices = np.argsort(prediction[0])[::-1][:3]  # Get indices of top 3 predictions
        top_classes = [class_names[i] for i in top_classes_indices]
        top_probs = [prediction[0][i] for i in top_classes_indices]

        # Display top predictions
        st.success("Top Predictions:")
        for i, (pred_class, prob) in enumerate(zip(top_classes, top_probs), 1):
            st.write(f"{i}. {pred_class}: {prob:.2f}")

        # Visualize predictions
        st.bar_chart({class_name: prob for class_name, prob in zip(top_classes, top_probs)})

    except (UnidentifiedImageError, Exception) as e:
        st.error("An error occurred during image processing or prediction. Please try again.")
        st.error(str(e))

# Instructions
st.markdown("""
### Instructions:
1. Upload the weather image (jpg, png, jpeg).
2. The model will predict the top 3 weather conditions along with their probabilities.
""")
