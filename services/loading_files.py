import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores import pinecone as langchain_pincone
from langchain_openai import OpenAIEmbeddings
from calc_tokens import num_tokens_from_string
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from get_embedding import get_embedding
from pinecone_init import initialize_pinecone
import dotenv

dotenv.load_dotenv()

print("SETTING UP PINECONE: ")

pc, pinecone_index = initialize_pinecone()

print("======================")

print("LOADING PAGES")
loader = PyPDFLoader("documents/disease_book.pdf")
pages = loader.load_and_split()

print(f"Number of documents: {len(pages)}")
print(f"Number of Characters in the first document: {len(pages[150].page_content)}")

print("===========================")

print("WORD EMBEDDINGS: ")

total_vector_count = pinecone_index.describe_index_stats()["total_vector_count"]
namespace_name = "book1"

for i in tqdm(range(total_vector_count, len(pages))):
    # first get metadata fields for this record
    metadata = {"page_content": pages[i].page_content}
    # create document embeddings
    embeds = get_embedding(pages[i].page_content)
    # add everything to pinecone
    vec = [{"id": f"{i}", "values": embeds, "metadata": metadata}]
    pinecone_index.upsert(vectors=vec, namespace=namespace_name)

print("===========================")
