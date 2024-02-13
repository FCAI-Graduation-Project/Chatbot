from controller import PlantAssistantController
from view import PlantView
from MlModel import MLModel
from ChatBotModel import ChatBotModel
import dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import OpenAI, ChatOpenAI
from CureDB import CureDB

dotenv.load_dotenv()

if __name__ == "__main__":
    ChatBotModel = ChatBotModel()
    MLModel = MLModel()
    view = PlantView()
    controller = PlantAssistantController(MLModel, ChatBotModel, view)
    controller.run()


# cureDB = CureDB()
# # text, num_tokens = cureDB.getCure("Apple", "Black Rot")
# text, num_tokens = cureDB.getCure("Tomato", "Late Blight")

# chat = ChatOpenAI(temperature=0)
# messages = [
#     SystemMessage(
#         content=f"""
#         You are a plant expert and you are given docs about a plant and it's disease,
#         Simplify the answer for newbie people in planting field.
#         You are supposed to answer user's question based on these docs.
#         If you didn't find answer in docs, say that you didn't find an answer.

#         ***

#         docs:
#         {text}
#         """
#     ),
# ]
# res = chat(messages).content
# print()
# print("ANSWER:")
# print()
# print(res)
# print()
