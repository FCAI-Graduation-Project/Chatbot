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

    def cureResponse(self, plantName, diseaseName, cureDocs, isHealthy):
        if isHealthy:
            return f"""
                    The plant captured in the photo is thriving and identified as a {plantName} Plant. and I'm pleased to inform you that it is free from any signs of disease.
                """

        template = """
                    The plant captured in the photo is thriving and identified as a {plantName} Plant. and I'm pleased to inform you that it is free from any signs of disease.

                    You are given a number of documents to get he best treatment for {diseaseName} in {plantName}.

                    ***

                    Docs:
                    {cureDocs}
                """

        prompt = PromptTemplate(
            template=template, input_variables=["plantName", "diseaseeName", "cureDocs"]
        )
        llm = OpenAI(temperature=0)
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        res = llm_chain.run(
            plantName=plantName, diseaseName=diseaseName, cureDocs=cureDocs
        )
        return res


"""
Create a comprehensive guide on the best treatment options for {diseaseName} in {plantName}, utilizing the provided documents ({cureDocs}).

Outline:

I. Introduction

A. Briefly explain the importance of treating {diseaseName} in {plantName}

B. Mention the relevance and reliability of the provided documents

II. Overview of {plantName} and {diseaseName}

A. Provide a brief description of {plantName}

B. Explain the characteristics and impact of {diseaseName} on {plantName}

III. Treatment Options

A. Document 1: {cureDoc1}

1. Summarize the key points and recommendations

2. Highlight any unique or innovative treatment methods or products mentioned

B. Document 2: {cureDoc2}

1. Summarize the key points and recommendations

2. Highlight any additional insights or alternative approaches suggested

C. Document 3: {cureDoc3}

1. Summarize the key points and recommendations

2. Discuss any scientific evidence or research cited in the document

D. Other Relevant Treatment Methods

1. Include any additional treatment options not covered in the provided documents

2. Provide a brief description and evaluation of each method's effectiveness

IV. Conclusion

A. Summarize the main findings and recommendations from the documents

B. Encourage readers to seek professional advice or consult with experts for personalized treatment plans

C. Reiterate the importance of timely and appropriate treatment for {diseaseName} in {plantName}

Note: Make sure to customize the outline by replacing {diseaseName}, {plantName}, and {cureDocs} with the specific variables provided.
"""
