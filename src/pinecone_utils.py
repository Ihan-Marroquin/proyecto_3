import os
from pinecone import Pinecone, ServerlessSpec
from typing import Optional

def get_pinecone_client(api_key: Optional[str] = None) -> Pinecone:
    api_key = api_key or os.getenv("PINECONE_API_KEY")
    return Pinecone(api_key=api_key)

def get_or_create_index(
    client: Pinecone,
    name: str,
    dimension: int,
    cloud: Optional[str] = None,
    region: Optional[str] = None
):
    existing = client.list_indexes().names()
    if name not in existing:
        spec = ServerlessSpec(
            cloud=cloud or os.getenv("PINECONE_CLOUD"),
            region=region or os.getenv("PINECONE_REGION")
        )
        client.create_index(
            name=name,
            dimension=dimension,
            metric="cosine",
            spec=spec
        )
    return client.Index(name=name)
