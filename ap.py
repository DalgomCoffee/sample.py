import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image,ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('finaltrain.h5')
  return model
  
model=load_model()
st.write("""
# Plant Leaf Detection System"""
)

file=st.file_uploader("Choose plant photo from computer",type=["jpg","png"])

def import_and_predict(image_data,model):
    size=(60,40)
    image=ImageOps.fit(image_data,size,Image.LANCZOS)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction
  
if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)
    class_names=['Rain','Shine','Cloudy','Sunrise']
    string="OUTPUT : "+class_names[np.argmax(prediction)]
    st.success(string)
