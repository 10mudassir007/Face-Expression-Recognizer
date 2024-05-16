import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np


model = tf.keras.models.load_model("main.keras")

uploaded_image = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])


if uploaded_image is not None:
    
    image = Image.open(uploaded_image)

    
    image_resized = image.resize((48, 48))
    image_gray = ImageOps.grayscale(image_resized)

    
    image_array = np.array(image_gray)

    
    image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    image_tensor = tf.expand_dims(image_tensor, axis=0)

    
    st.image(image, caption="Uploaded Image", use_column_width=True)


    prediction = model.predict(image_tensor)
    predicted = tf.argmax(prediction,axis=1)
    emotion = int(predicted)
    if emotion == 0:
        st.write("Emotion: Angry")
    elif emotion == 1:
       st.write("Emotion: Fear")
    elif emotion == 2:
        st.write("Emotion: Happy")
    elif emotion == 3:
        st.write("Emotion: Neutral")
    elif emotion == 4:
        st.write("Emotion: Sad")
    elif emotion == 5:
        st.write("Emotion: Surprise")
