import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image as keras_image

classes_dict = [
    "Apple__black_rot",
    "Apple__healthy",
    "Apple__rust",
    "Apple__scab",
    "Cassava__bacterial_blight",
    "Cassava__brown_streak_disease",
    "Cassava__green_mottle",
    "Cassava__healthy",
    "Cassava__mosaic_disease",
    "Cherry__healthy",
    "Cherry__powdery_mildew",
    "Chili__healthy",
    "Chili__leaf_curl",
    "Chili__leaf_spot",
    "Chili__whitefly",
    "Chili__yellowish",
    "Coffee__cercospora_leaf_spot",
    "Coffee__healthy",
    "Coffee__red_spider_mite",
    "Coffee__rust",
    "Corn__common_rust",
    "Corn__gray_leaf_spot",
    "Corn__healthy",
    "Corn__northern_leaf_blight",
    "Cucumber__diseased",
    "Cucumber__healthy",
    "Gauva__diseased",
    "Gauva__healthy",
    "Grape__black_measles",
    "Grape__black_rot",
    "Grape__healthy",
    "Grape__leaf_blight_(isariopsis_leaf_spot)",
    "Jamun__diseased",
    "Jamun__healthy",
    "Lemon__diseased",
    "Lemon__healthy",
    "Mango__diseased",
    "Mango__healthy",
    "Peach__bacterial_spot",
    "Peach__healthy",
    "Pepper_bell__bacterial_spot",
    "Pepper_bell__healthy",
    "Pomegranate__diseased",
    "Pomegranate__healthy",
    "Potato__early_blight",
    "Potato__healthy",
    "Potato__late_blight",
    "Rice__brown_spot",
    "Rice__healthy",
    "Rice__hispa",
    "Rice__leaf_blast",
    "Rice__neck_blast",
    "Soybean__bacterial_blight",
    "Soybean__caterpillar",
    "Soybean__diabrotica_speciosa",
    "Soybean__downy_mildew",
    "Soybean__healthy",
    "Soybean__mosaic_virus",
    "Soybean__powdery_mildew",
    "Soybean__rust",
    "Soybean__southern_blight",
    "Strawberry___leaf_scorch",
    "Strawberry__healthy",
    "Sugarcane__bacterial_blight",
    "Sugarcane__healthy",
    "Sugarcane__red_rot",
    "Sugarcane__red_stripe",
    "Sugarcane__rust",
    "Tea__algal_leaf",
    "Tea__anthracnose",
    "Tea__bird_eye_spot",
    "Tea__brown_blight",
    "Tea__healthy",
    "Tea__red_leaf_spot",
    "Tomato__bacterial_spot",
    "Tomato__early_blight",
    "Tomato__healthy",
    "Tomato__late_blight",
    "Tomato__leaf_mold",
    "Tomato__mosaic_virus",
    "Tomato__septoria_leaf_spot",
    "Tomato__spider_mites_(two_spotted_spider_mite)",
    "Tomato__target_spot",
    "Tomato__yellow_leaf_curl_virus",
    "Wheat__brown_rust",
    "Wheat__healthy",
    "Wheat__septoria",
    "Wheat__yellow_rust",
]

def getDisease(img):
    model = load_model("model.h5")
    img = img.resize((128, 128))
    img_array = keras_image.img_to_array(img)  # Use the renamed module
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    predictions = model.predict(img_array)

    class_index = np.argmax(predictions[0])
    predicted_class = classes_dict[class_index]
    return predicted_class, np.max(predictions[0])

def general_questions():
    question = st.text_input("Ask your question:")
    if st.button("Ask"):
        # Here you can implement a function to process the question and generate the answer
        # For now, let's just echo back the question as the answer
        st.write("Your question:", question)
        # st.write("Your answer:" , ans)

def check_for_disease():
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        if st.button('Predict'):
            prediction, confidence = getDisease(Image.open(uploaded_file))
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
            st.write(f'Prediction: {prediction}, Confidence: {confidence}')

st.title('🌱 Smart Plant Care🌱')

task_option = st.radio("Choose a task:", ["Ask general questions", "Check for disease"])

if task_option == "Ask general questions":
    general_questions()
else:
    check_for_disease()
