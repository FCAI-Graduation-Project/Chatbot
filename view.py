from flask import render_template, jsonify
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
import dotenv
dotenv.load_dotenv()


class PlantView:
    def renderHome(self):
        return render_template("home.html")

    def renderError(self, message):
        return jsonify({'error': message}), 400
