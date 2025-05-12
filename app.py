import streamlit as st
from dotenv import load_dotenv
import os

from src.loader import load_and_split_documents
from src.pinecone_utils import get_pinecone_client, get_or_create_index
from src.chain import build_vectorstore, get_qa_chain

load_dotenv()
st.set_page_config(page_title="Asistente Técnico", layout="wide")

def setup():
   
    pc = get_pinecone_client()

    index = get_or_create_index(
        client=pc,
        name=os.getenv("PINECONE_INDEX_NAME"),
        dimension=int(os.getenv("PINECONE_DIMENSION", 1536)),
        cloud=os.getenv("PINECONE_CLOUD", None),
        region=os.getenv("PINECONE_REGION", None)
    )

    docs = load_and_split_documents("data/documentos")
    vectorstore = build_vectorstore(docs)
    return get_qa_chain(vectorstore)

if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = setup()

st.title("Asistente de Consulta Técnica")
query = st.text_input("Ingresa tu pregunta:")
if st.button("Preguntar") and query:
    with st.spinner("Generando respuesta..."):
        response = st.session_state.qa_chain({"query": query})
        st.subheader("Respuesta:")
        st.write(response['result'])
        st.subheader("Contexto utilizado:")
        for doc in response['source_documents']:
            st.write(doc.page_content)
