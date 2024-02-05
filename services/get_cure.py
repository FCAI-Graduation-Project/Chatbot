from pinecone_init import initialize_pinecone
import dotenv
from get_embedding import get_embedding
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from calc_tokens import num_tokens_from_string

dotenv.load_dotenv()


def get_cure():
    # llm = ChatOpenAI(temperature=0)
    # chain = load_qa_chain(llm, chain_type="stuff")
    query = "potato plant Managment of black dot disease?"
    query_vec = get_embedding(query)
    _, pinecone_index = initialize_pinecone()

    query_output = pinecone_index.query(vector=query_vec, top_k=5, namespace="book1")

    matching_pages_ids = []
    for vec in query_output["matches"]:
        matching_pages_ids.append(vec["id"])

    matching_pages_content = pinecone_index.fetch(
        ids=matching_pages_ids, namespace="book1"
    )

    sum_text = ""

    for vec_id in matching_pages_content.to_dict()["vectors"].keys():
        page_content = matching_pages_content.to_dict()["vectors"][vec_id]["metadata"][
            "page_content"
        ]
        print("=========================================")
        print(f"ID: {vec_id}")
        print("----")
        print()
        print(page_content)
        print()
        print("=========================================")

        sum_text += page_content

    print(f"Tokens: {num_tokens_from_string(sum_text)}")


get_cure()
