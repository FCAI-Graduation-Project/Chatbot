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

print("LOADING PAGES")
loader = PyPDFLoader("documents/disease_book.pdf")
pages = loader.load_and_split()

print(f"Number of documents: {len(pages)}")
print(f"Number of Characters in the first document: {len(pages[150].page_content)}")

print("===========================")

print("WORD EMBEDDINGS: ")

for i in tqdm(range(0, len(pages))):
    # first get metadata fields for this record
    metadata = {"page_content": pages[i].page_content}
    # create document embeddings
    embeds = get_embedding(pages[i].page_content)
    # get IDs
    id = i
    # add everything to pinecone
    vec = [{"id": f"{i}", "values": embeds, "metadata": metadata}]
    print("Vector")
    pinecone_index.upsert(vectors=vec, namespace="book1")

print("===========================")
