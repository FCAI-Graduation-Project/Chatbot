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

print("SETTING UP: ")
print("PINECONE: ")

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
# faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
# faiss_index
# docsearch = langchain_pincone.Pinecone.from_texts(
#     [page.page_content for page in pages[:10]],
#     OpenAIEmbeddings(),
#     index_name=pinecone_index_name,
# )

# embeddings = []

# pinecone_index.upsert(embeddings)


# for i in tqdm(range(0, len(pages))):
for i in tqdm(range(0, 10)):
    # first get metadata fields for this record
    metadata = pages[i].page_content
    # create document embeddings
    embeds = get_embedding(pages[i].page_content)
    print(embeds)
    # get IDs
    id = i
    # add everything to pinecone
    vec = [{"id": f"{i}", "values": embeds, "metadata": metadata}]
    print("Vector")
    print(vec)
    pinecone_index.upsert(vectors=vec, namespace="book1")

print("===========================")

print("DOING SIMILARITY SEACRH")
# docs = docsearch.similarity_search("how to manage Tomoato with early blight disease?")

# for doc in docs:
#     print(str(doc.metadata["page"]) + ":", doc.page_content)

print("===========================")
