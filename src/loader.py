import os
import logging
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def load_and_split_documents(
    path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    encodings: list[str] = ["utf-8", "latin-1", "cp1252"]
) -> list[Document]:
    documents: list[Document] = []
    for fname in os.listdir(path):
        full_path = os.path.join(path, fname)
        content = None
        
        if not fname.lower().endswith(".txt"):
            logger.info(f"Saltando {fname} (extensión no soportada)")
            continue

        for enc in encodings:
            try:
                loader = TextLoader(full_path, encoding=enc)
                docs = loader.load()
                content = docs
                break
            except Exception as e:
                logger.warning(f"No pudo leer {fname} con {enc}: {e}")
        
        if content is None:
            logger.error(f"Falló la carga de {fname} con todas las codificaciones, se omite.")
            continue
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        try:
            split_docs = splitter.split_documents(content)
            documents.extend(split_docs)
        except Exception as e:
            logger.error(f"Error fragmentando {fname}: {e}")
            continue

    return documents
