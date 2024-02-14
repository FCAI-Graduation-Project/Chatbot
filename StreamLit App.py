import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
# HERE : make my own prediction function 
def predict_disease(image):
    pass


st.title('🌱Plant and Disease Detection🌱')
uploaded_file = st.file_uploader("Enter an Image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     image = tf.image.decode_image(uploaded_file.getvalue(), channels=3)
#     st.image(image, caption='Uploaded Image', use_column_width=True)
    

#     # Button to trigger prediction
#     if st.button('Predict'):
#         # Make predictions
#         prediction = predict_disease(image)
        
#         # Display the prediction
#         st.write(f'Prediction: {prediction}')
