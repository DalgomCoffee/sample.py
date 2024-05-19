import streamlit as st
import tensorflow as tf
import cv2
import io


def load_model():
  model=tf.keras.models.load_model('bestmodel.h5')
  return model

model=load_model('bestmodel.h5')
st.write("""
# Plant Leaf Detection System"""
)

uploaded_file=st.file_uploader("Choose weather photo ",type=["jpg","png","jpeg"])

from PIL import Image,ImageOps
import numpy as np

def import_and_predict(image_data, model):
    file.seek(0)
    image = Image.open(file)
    image = image.resize((60, 40))
    img = np.array(image)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    img = np.reshape(img, (1, 2160, 3, 1))
    prediction = model.predict(img)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(file,model)
    class_names=['Rain','Shine','Cloudy','Sunrise']
    string="OUTPUT : "+class_names[np.argmax(prediction)]
    st.success(string)
