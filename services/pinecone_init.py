import os
from pinecone import Pinecone


def initialize_pinecone():
    pinecone = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENV"),
    )
    pinecone_index = pinecone.Index(os.getenv("PINECONE_INDEX"))

    return pinecone, pinecone_index
