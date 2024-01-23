from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model='gpt-3.5-turbo')

res = llm.invoke("how can langsmith help with testing?")

print(res.content)


