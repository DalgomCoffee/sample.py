import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('finaltrain.h5')
    return model

model = load_model()

# Display the model's input shape for debugging
st.write(f'Model input shape: {model.input_shape}')

st.write("""
# Plant Leaf Detection System
""")

file = st.file_uploader("Choose plant photo from computer", type=["jpg", "png"])

def import_and_predict(image_data, model):
    # Determine the input shape expected by the model
    input_shape = model.input_shape[1:]  # Exclude the batch dimension
    
    # If the input shape is flattened (e.g., (2400,)), we need to handle it differently
    if len(input_shape) == 1:
        expected_pixels = input_shape[0]
        width = height = int(np.sqrt(expected_pixels // 3))  # Assuming 3 channels (RGB)
        size = (width, height)
        image = ImageOps.fit(image_data, size, Image.LANCZOS)
        img = np.asarray(image)
        img = img / 255.0  # Normalize the image
        img_reshape = img.reshape((1, -1))  # Flatten the image
    else:
        # If the input shape is not flattened, assume it to be (height, width, channels)
        size = input_shape[:2]
        image = ImageOps.fit(image_data, size, Image.LANCZOS)
        img = np.asarray(image)
        img = img / 255.0  # Normalize the image
        img_reshape = img[np.newaxis, ...]  # Add batch dimension
    
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['Rain', 'Shine', 'Cloudy', 'Sunrise']
    string = "OUTPUT : " + class_names[np.argmax(prediction)]
    st.success(string)
