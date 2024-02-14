from getClassByIndex import getClassByIndex
import tensorflow as tf
import numpy as np
import dotenv

dotenv.load_dotenv()


class MLModel:

    def __init__(self):
        pass

    def getDisease(self, img):
        model = tf.keras.models.load_model("model.h5")

        # Prepare the image for prediction
        finalImage = np.array(np.expand_dims(img, axis=0), dtype="float64")

        # Normalize the image
        finalImage /= 255.0

        # Make predictions
        predictions = model.predict(finalImage)

        # Get the class name
        classIndex = np.argmax(predictions)
        confidence = np.max(predictions)
        imgClass = getClassByIndex(classIndex)

        print("CLASS INDEX")
        print(classIndex)

        print("Class name\n")
        print(imgClass)

        print("softmax value")
        print(confidence)

        return imgClass
