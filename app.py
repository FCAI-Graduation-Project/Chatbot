from controller import PlantAssistantController
from view import PlantView
from MlModel import MLModel
from ChatBotModel import ChatBotModel
import dotenv

dotenv.load_dotenv()

if __name__ == "__main__":
    ChatBotModel = ChatBotModel()
    MLModel = MLModel()
    view = PlantView()
    controller = PlantAssistantController(MLModel, ChatBotModel, view)
    controller.run()
