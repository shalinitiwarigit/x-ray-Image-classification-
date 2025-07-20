import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

model = load_model('model.h5')

st.title("🩻 X-ray Image Classification")
st.write("Upload a chest Image an check whether the report shows sign of **NORMAL** OR **PNEUMONIA** using Deep Learning")
st.markdown("Check your X-ray report")
uploaded_file = st.file_uploader("📤 Upload an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')


    img=img.resize((150,150))
    img_array=image.img_to_array(img)
    img_array=np.expand_dims(img_array,axis=0)
    img_array/=255

    prediction=model.predict(img_array)


    #show result
    if prediction[0][0]>0.5:
        st.error("🔴 Result is **PNEUMONIA**")

    else:
        st.success("🟢 Result is **NORMAL**")
