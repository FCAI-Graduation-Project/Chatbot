import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image as keras_image
from ChatBotModel import ChatBotModel
from getClassByIndex import getClassByIndex
import dotenv
dotenv.load_dotenv()


def getDisease(img):
    model = load_model("TestModel.h5")
    img = img.resize((224, 224))
    img_array = keras_image.img_to_array(img)  # Use the renamed module
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    predictions = model.predict(img_array)

    class_index = np.argmax(predictions[0])
    predicted_class = getClassByIndex(class_index)
    return predicted_class, np.max(predictions[0])

def askGeneralQuestion()-> None:
    question = st.text_input("Ask your question:")
    # print(question)
    if st.button("Ask"):
        # st.write("Your question:", question)
        # question = question.title()
        # st.write("Test question" , question)
        botModel = ChatBotModel()
        ret = botModel.getResponse(question).strip()
        # st.write("Test ret" , ret)
        category = ["general question", "cure of disease", "other"]
        ans = "Error"
        if ret == category[0]:
            ans = botModel.generalQuestion(question)
        else:
            ans = botModel.other()
        st.write(ans)
def cureOfPlant() ->None:
    plantName = st.text_input("Enter your plant name")
    diseaseName = st.text_input("Enter the disease name")
    if st.button("Ask"):
        botModel = ChatBotModel()
        # if st.button("Send"):
        ans = botModel.cureOfDisease(plantName , diseaseName)
        finalSteps = botModel.summarize(ans)
        ans = finalSteps
        st.write(ans)
    
def check_for_disease() -> None:
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        if st.button('Predict'):
            prediction, confidence = getDisease(Image.open(uploaded_file))
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
            st.write(f'Prediction: {prediction}, Confidence: {confidence}')

st.title('🌱 Smart Plant Care🌱')

task_option = st.radio("Choose a task:", ["Ask general questions","cure of plant","Check for disease"])

if task_option == "Ask general questions":
    askGeneralQuestion()
elif task_option == "cure of plant":
    cureOfPlant()
else:
    check_for_disease()


# if st.button("Check for disease"):
#     uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
#     if uploaded_file is not None:
#         if st.button('Predict'):
#             prediction, confidence = getDisease(Image.open(uploaded_file))
#             st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
#             st.write(f'Prediction: {prediction}, Confidence: {confidence}')
    
# if st.button("Ask a question"):
#     response = askQuestion()
#     st.write(response)
