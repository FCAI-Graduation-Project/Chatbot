from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import OpenAI, ChatOpenAI
import os
import tensorflow as tf
import cv2
from keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import dotenv
dotenv.load_dotenv()

labels = ['Tomato__septoria_leaf_spot',
          'Tea__anthracnose',
          'Rice__hispa',
          'Soybean__diabrotica_speciosa',
          'Tomato__healthy',
          'Gauva__diseased',
          'Wheat__yellow_rust',
          'Soybean__downy_mildew',
          'Pomegranate__diseased',
          'Apple__rust',
          'Tea__algal_leaf',
          'Pomegranate__healthy',
          'Tomato__early_blight',
          'Peach__bacterial_spot',
          'Soybean__healthy',
          'Potato__healthy',
          'Cherry__healthy',
          'Apple__healthy',
          'Rice__neck_blast',
          'Wheat__septoria',
          'Jamun__healthy',
          'Strawberry___leaf_scorch',
          'Coffee__rust',
          'Mango__healthy',
          'Mango__diseased',
          'Apple__black_rot',
          'Soybean__powdery_mildew',
          'Lemon__healthy',
          'Pepper_bell__healthy',
          'Strawberry__healthy',
          'Cassava__healthy',
          'Corn__healthy',
          'Jamun__diseased',
          'Corn__common_rust',
          'Tomato__yellow_leaf_curl_virus',
          'Coffee__cercospora_leaf_spot',
          'Grape__black_measles',
          'Rice__healthy',
          'Tea__healthy',
          'Soybean__caterpillar',
          'Grape__leaf_blight_(isariopsis_leaf_spot)',
          'Coffee__red_spider_mite',
          'Soybean__rust',
          'Tomato__target_spot',
          'Tomato__bacterial_spot',
          'Sugarcane__bacterial_blight',
          'Tea__brown_blight',
          'Sugarcane__healthy',
          'Cucumber__healthy',
          'Chili__whitefly',
          'Cucumber__diseased',
          'Soybean__bacterial_blight',
          'Potato__early_blight',
          'Chili__healthy',
          'Tea__red_leaf_spot',
          'Corn__gray_leaf_spot',
          'Grape__black_rot',
          'Potato__late_blight',
          'Sugarcane__red_stripe',
          'Cassava__mosaic_disease',
          'Cassava__green_mottle',
          'Tomato__late_blight',
          'Pepper_bell__bacterial_spot',
          'Chili__leaf spot',
          'Rice__brown_spot',
          'Lemon__diseased',
          'Cassava__brown_streak_disease',
          'Wheat__brown_rust',
          'Tomato__spider_mites_(two_spotted_spider_mite)',
          'Sugarcane__rust',
          'Coffee__healthy',
          'Tomato__leaf_mold',
          'Cherry__powdery_mildew',
          'Apple__scab',
          'Soybean__southern_blight',
          'Rice__leaf_blast',
          'Corn__northern_leaf_blight',
          'Gauva__healthy',
          'Peach__healthy',
          'Soybean__mosaic_virus',
          'Chili__yellowish',
          'Cassava__bacterial_blight',
          'Tea__bird_eye_spot',
          'Wheat__healthy',
          'Sugarcane__red_rot',
          'Chili__leaf curl',
          'Grape__healthy',
          'Tomato__mosaic_virus']


class MLModel:

    def __init__(self):
        pass

    def getDisease(self, img):
        model = tf.keras.models.load_model("D:/FCAI/pykeee/chat bot/model.h5")

        # Load and preprocess the image using OpenCV
        imgArray = tf.keras.preprocessing.image.img_to_array(img)
        imgArray = preprocess_input(imgArray)

        # Make predictions
        predictions = model.predict(
            np.expand_dims(imgArray, axis=0))  # Add an extra dimension to match the model input shape

        # Get the index of the class with the highest probability using argmax
        classIndex = np.argmax(predictions)
        return labels[classIndex]
