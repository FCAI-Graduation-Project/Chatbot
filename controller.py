from flask import Flask, request
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
import dotenv

dotenv.load_dotenv()


class PlantAssistantController:
    category = ["general question", "cure of disease", "other"]

    def __init__(self, MLModel, ChatBotModel, View):
        self.MlModel = MLModel
        self.ChatBotModel = ChatBotModel
        self.view = View
        self.app = Flask(__name__)
        self.setupRoutes()

    def setupRoutes(self):
        self.app.route('/')(self.home)
        self.app.route('/getQuestion', methods=['GET'])(self.getQuestion)

    def run(self):
        self.app.run(debug=True, port=9000)

    def home(self):
        return self.view.renderHome()

    def getQuestion(self):
        if request.method == 'GET':
            userQuestion = request.args.get('user_question')
            if userQuestion:
                ret = self.ChatBotModel.getResponse(userQuestion).strip()
                print(ret.strip())
                if ret == self.category[0]:
                    return self.ChatBotModel.generalQuestion(userQuestion)
                elif ret == self.category[1]:
                    plantName = "Apple"
                    diseaseName = "black rot"
                    return self.ChatBotModel.cureOfDisease(plantName, diseaseName)
                else:
                    return self.ChatBotModel.other()
            else:
                return self.view.renderError('Empty question provided')
        else:
            return self.view.renderHome()
