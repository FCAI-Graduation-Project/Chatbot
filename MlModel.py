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
import matplotlib.pyplot as plt

dotenv.load_dotenv()

labels = ['Apple__black_rot', 'Apple__healthy', 'Apple__rust', 'Apple__scab', 'Cassava__bacterial_blight', 'Cassava__brown_streak_disease', 'Cassava__green_mottle', 'Cassava__healthy', 'Cassava__mosaic_disease', 'Cherry__healthy', 'Cherry__powdery_mildew', 'Chili__healthy', 'Chili__leaf_curl', 'Chili__leaf_spot', 'Chili__whitefly', 'Chili__yellowish', 'Coffee__cercospora_leaf_spot', 'Coffee__healthy', 'Coffee__red_spider_mite', 'Coffee__rust', 'Corn__common_rust', 'Corn__gray_leaf_spot', 'Corn__healthy', 'Corn__northern_leaf_blight', 'Cucumber__diseased', 'Cucumber__healthy', 'Gauva__diseased', 'Gauva__healthy', 'Grape__black_measles', 'Grape__black_rot', 'Grape__healthy', 'Grape__leaf_blight_(isariopsis_leaf_spot)', 'Jamun__diseased', 'Jamun__healthy', 'Lemon__diseased', 'Lemon__healthy', 'Mango__diseased', 'Mango__healthy', 'Peach__bacterial_spot', 'Peach__healthy', 'Pepper_bell__bacterial_spot', 'Pepper_bell__healthy', 'Pomegranate__diseased', 'Pomegranate__healthy', 'Potato__early_blight', 'Potato__healthy', 'Potato__late_blight', 'Rice__brown_spot', 'Rice__healthy', 'Rice__hispa', 'Rice__leaf_blast', 'Rice__neck_blast', 'Soybean__bacterial_blight', 'Soybean__caterpillar', 'Soybean__diabrotica_speciosa', 'Soybean__downy_mildew', 'Soybean__healthy', 'Soybean__mosaic_virus', 'Soybean__powdery_mildew', 'Soybean__rust', 'Soybean__southern_blight', 'Strawberry___leaf_scorch', 'Strawberry__healthy', 'Sugarcane__bacterial_blight', 'Sugarcane__healthy', 'Sugarcane__red_rot', 'Sugarcane__red_stripe', 'Sugarcane__rust', 'Tea__algal_leaf', 'Tea__anthracnose', 'Tea__bird_eye_spot', 'Tea__brown_blight', 'Tea__healthy', 'Tea__red_leaf_spot', 'Tomato__bacterial_spot', 'Tomato__early_blight', 'Tomato__healthy', 'Tomato__late_blight', 'Tomato__leaf_mold', 'Tomato__mosaic_virus', 'Tomato__septoria_leaf_spot', 'Tomato__spider_mites_(two_spotted_spider_mite)', 'Tomato__target_spot', 'Tomato__yellow_leaf_curl_virus', 'Wheat__brown_rust', 'Wheat__healthy', 'Wheat__septoria', 'Wheat__yellow_rust']

class MLModel:

    def __init__(self):
        pass

    def getDisease(self, img):
        model = tf.keras.models.load_model("model.h5")

        # Load and preprocess the image using OpenCV
        # imgArray = tf.keras.preprocessing.image.img_to_array(img)
        # imgArray = preprocess_input(imgArray)
        # plt.imshow(imgArray)
        
        # Display the original image
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title('Original Image')

        # Prepare the image for prediction
        finalImage = np.expand_dims(img, axis=0)

        # Display the preprocessed image
        plt.subplot(1, 2, 2)
        plt.imshow(finalImage[0])
        plt.title('Preprocessed Image')

        plt.show()
        
        # Make predictions
        predictions = model.predict(finalImage)
        
        classIndex = np.argmax(predictions)
        confidence = np.max(predictions)
        print("CLASS INDEX")
        print(classIndex , type(classIndex))
        print("Class name\n")
        print(labels[classIndex])
        print("softmax value")
        print(confidence)
        return labels[classIndex]
