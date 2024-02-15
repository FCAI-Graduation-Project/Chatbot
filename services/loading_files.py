from langchain_community.document_loaders import PyPDFLoader
from tqdm.auto import tqdm
from get_embedding import get_embedding
from pinecone_init import initialize_pinecone
import dotenv

dotenv.load_dotenv()

print("SETTING UP PINECONE: ")

pc, pinecone_index = initialize_pinecone()

print("======================")

# print("LOADING PAGES")
# loader = PyPDFLoader("documents/disease_book.pdf")
# pages = loader.load_and_split()

# print(f"Number of documents: {len(pages)}")
# print(f"Number of Characters in the first document: {len(pages[150].page_content)}")

# for page in pages[0:65]:
#     print("=================")
#     print(page.page_content)
#     print("=================")


# total_vector_count = pinecone_index.describe_index_stats()["total_vector_count"]
namespace_name = "book1"

# this list map function just convetts list of integers to list of strings, list is from 64 to 658
vecs_map = pinecone_index.fetch(
    # list(map(str, range(64, 358))), namespace=namespace_name
    list(map(str, range(358, 658))),
    namespace=namespace_name,
)["vectors"]

# print(vecs["vectors"])

vec_list = []
for key in vecs_map.keys():
    vec_list.append(vecs_map[key])

pinecone_index.upsert(vectors=vec_list, namespace="book2")


# print("===========================")

# print("WORD EMBEDDINGS: ")

# total_vector_count = pinecone_index.describe_index_stats()["total_vector_count"]
# namespace_name = "book2"

# for i in tqdm(range(total_vector_count, len(pages))):
#     # first get metadata fields for this record
#     metadata = {"page_content": pages[i].page_content}
#     # create document embeddings
#     embeds = get_embedding(pages[i].page_content)
#     # add everything to pinecone
#     vec = [{"id": f"{i}", "values": embeds, "metadata": metadata}]
#     pinecone_index.upsert(vectors=vec, namespace=namespace_name)

# print("===========================")
