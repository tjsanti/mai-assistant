from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings


@lru_cache
def get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name)

