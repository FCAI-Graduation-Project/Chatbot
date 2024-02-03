from openai import OpenAI


def get_embedding(text: str, model: str = "text-embedding-3-small"):
    client = OpenAI()

    return client.embeddings.create(input=[text], model=model).data[0].embedding
