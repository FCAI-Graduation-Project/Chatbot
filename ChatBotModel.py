from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import OpenAI, ChatOpenAI
import os
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
from CureDB import CureDB
import dotenv

dotenv.load_dotenv()


class ChatBotModel:

    def __init__(self):
        pass

    def getResponse(self, user_question):
        template = f"""
                    You are a plant assistant, and your task is to classify user questions. 
                    Please categorize the following question into one of the categories from this list: {user_question} - ["general question", "cure of disease", "other"].
                    For example:
                    - If the question is "What is the Apple?","How can I plant a Tomato?", it should be classified as 'general question'.
                    - If the question is "What is the apple black rot?", it should be classified as 'general question'.
                    - If the question is "What is the cure of Apple black rot?", it should be classified as 'cure of disease'.
                    - If the question is "What is the football? Who is Mohamed Salah?", it should be classified as 'other'.
                    just return the value os a string like 'general question'
                """
        prompt = PromptTemplate(template=template, input_variables=["user_question"])
        llm = OpenAI(temperature=0)
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        ret = llm_chain.run(user_question=user_question)
        return ret

    def generalQuestion(self, user_question):
        chat = ChatOpenAI(temperature=0)
        messages = [
            SystemMessage(
                content="""
                        You are a helpful AI Plant assistant that answers the questions about Plants and its fields
                         but any other field outside the plant or the agriculture don't response and say 
                         Sorry I'm an AI plant assistant.
                         """
            ),
            HumanMessage(content=user_question),
        ]
        res = chat(messages).content
        return res

    def other(self):
        return """
                Sorry, I'm an AI plant assistant. 
                How can I assist you with this categories : ["general question", "cure of disease", "other"]
                """

    def cureOfDisease(self, plantName, diseaseName):
        cure = CureDB()
        matchingPagesContent, total_tokens = cure.getCure(plantName, diseaseName)
        return matchingPagesContent
