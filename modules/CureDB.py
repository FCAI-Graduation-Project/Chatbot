import os
import dotenv
import tiktoken
from openai import OpenAI
from pinecone import Pinecone

dotenv.load_dotenv()


class CureDB:
    def __init__(self) -> None:
        pass

    def _initilializePinecone(self):
        """
        Creates an object of pinecone to access my hosted vectors
        """
        pinecone = Pinecone(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENV"),
        )
        pinecone_index = pinecone.Index(os.getenv("PINECONE_INDEX"))

        return pinecone, pinecone_index

    def __getEmbeddings(
        self, text: str, model: str = "text-embedding-3-small"
    ) -> list[float]:
        """
        Return the Embeddings of a text
        """
        client = OpenAI()

        return client.embeddings.create(input=[text], model=model).data[0].embedding

    def __calcTokens(self, string: str, encoding_name: str = "cl100k_base") -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def getCure(self, label: str):
        query = "potato plant Managment of late blight disease?"
        query_vec = self.__getEmbedding(query)
        _, pinecone_index = self.__initializePinecone()

        query_output = pinecone_index.query(
            vector=query_vec, top_k=5, namespace="book1"
        )

        matching_pages_ids = []
        for vec in query_output["matches"]:
            matching_pages_ids.append(vec["id"])

        matching_pages_content = pinecone_index.fetch(
            ids=matching_pages_ids, namespace="book1"
        )

        sum_text = ""

        for vec_id in matching_pages_content.to_dict()["vectors"].keys():
            page_content = matching_pages_content.to_dict()["vectors"][vec_id][
                "metadata"
            ]["page_content"]
            print("=========================================")
            print(f"ID: {vec_id}")
            print("----")
            print()
            print(page_content)
            print()
            print("=========================================")

            sum_text += page_content

        print(f"Tokens: {self.__calcTokens(sum_text)}")
