import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores import pinecone as langchain_pincone
from langchain_openai import OpenAIEmbeddings
from calc_tokens import num_tokens_from_string
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from get_embedding import get_embedding
import dotenv

dotenv.load_dotenv()

print("SETTING UP PINECONE: ")

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV"),
)
pinecone_index = pc.Index(os.getenv("PINECONE_INDEX"))

print("======================")

print("DELETING NAMESPACE: ")

pinecone_index.delete(delete_all=True, namespace="book2")

print("======================")
