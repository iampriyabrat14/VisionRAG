from openai import OpenAI

client = OpenAI()

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


def embed_text(text: str) -> list[float]:
    text = text.replace("\n", " ").strip()
    response = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return response.data[0].embedding


def embed_batch(texts: list[str]) -> list[list[float]]:
    cleaned = [t.replace("\n", " ").strip() for t in texts]
    response = client.embeddings.create(input=cleaned, model=EMBEDDING_MODEL)
    return [item.embedding for item in response.data]
