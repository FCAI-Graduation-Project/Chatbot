from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain


memory = [
    (
        "system",
        """
        You are a chatbot that only answers questions about plants and agriculture domain, if the user tries
        to ask anything else don't answer him.
        """,
    ),
]


def master_chatbot(userMessage):
    chat_res = userMessage

    # llm = ChatOpenAI()
    # chat_template = ChatPromptTemplate.from_messages(memory)
    # Notice that we `return_messages=True` to fit into the MessagesPlaceholder
    # Notice that `"chat_history"` aligns with the MessagesPlaceholder name.
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)

    # conversation({"question": "hi"})

    return chat_res


# asks a genreal question     ->
# has a plant and wants to know the disease           -> send to model
# wants the cure                   ->        fine tune model a
