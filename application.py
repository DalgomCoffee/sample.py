import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, UnidentifiedImageError

model = load_model('finaltrain.h5')
class_names = ['Cloudy', 'Shine', 'Rain', 'Sunrise']

def preprocess_image(image, target_size=(60, 40)):
    image = image.resize(target_size)
    image = image.convert('L')  
    image = np.array(image)
    image = image / 255.0  
    image = image.flatten()  
    image = np.expand_dims(image, axis=-0)  
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
1. Upload a weather image (jpg, png, jpeg).
2. The prediction will be displayed.
""")

import requests
from tkinter import *


root=Tk()
root.geometry("500x500")
root.config(bg="black")
root.title("weather app")

enter_city=Entry(root,font="lucinda 20 bold")
enter_city.pack()

def search():
    my_url="http://api.openweathermap.org/data/2.5/weather"
    api_key="e170aa40f3631125e4a20447689e9eb7"
    parameters={
        "q":enter_city.get(),
        "appid":api_key

    }

    response=requests.get(url=my_url,params=parameters)
    temp_data_kelvin=response.json()["main"]["temp"]
    temp_data_celsius=temp_data_kelvin-273.15

    text_temp.config(text=f" current temparature \n of {enter_city.get()} is :{int(temp_data_celsius)} degree C")


b1=Button(root,text="seach location",font="lucinda 30 bold",fg="black",bg="black",command=search)
b1.pack(padx=20,pady=20)

text_temp = Label(text=".....",font="lucinda 25 bold")
text_temp.pack(padx=20, pady=20)

root.mainloop()
