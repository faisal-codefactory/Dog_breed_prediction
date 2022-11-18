import numpy as np
import streamlit as st
import cv2
from tensorflow.keras.models import load_model

# Loading the Model
model = load_model('dog_breed.h5')

# Name of Classes
CLASS_NAMES = ['Scottish Deerhound', 'Maltese Dog', 'Bernese Mountain Dog']

# Setting Title of App
st.title('Dog Breed Prediction')
st.markdown('Upload an image of the dog')

# Uploading the dog image
dog_image = st.file_uploader('Choose and image...', type='png')
submit = st.button('Predict')
if submit:
    if dog_image is not None:
        # conver the file to an openvc image
        file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode (file_bytes, 1)

        # displaying the image
        st.image(opencv_image, channels = 'BGR')
        # Resize image
        opencv_image = cv2.resize(opencv_image, (224,224))
        # convert to 4D
        opencv_image.shape = (1,224,224,3)
        # make prediction
        y_pred = model.predict(opencv_image)
        st.title(str(f"The Dog breed is {CLASS_NAMES[np.argmax(y_pred)]}"))

