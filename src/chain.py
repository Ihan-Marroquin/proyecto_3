import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

def build_vectorstore(docs):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    vectorstore = PineconeVectorStore.from_texts(
        texts=[doc.page_content for doc in docs],
        embedding=embeddings,
        metadatas=[doc.metadata for doc in docs],
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        namespace=None
    )

    return vectorstore

def get_qa_chain(vectorstore):
    llm = ChatOpenAI(
        temperature=0.0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True
    )
    return qa
