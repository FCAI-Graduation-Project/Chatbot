from openai import OpenAI


def get_embedding(embeds: str, model: str = "text-embedding-3-small"):
    client = OpenAI()

    return client.embeddings.create(input=embeds, model=model).data[0].embedding
