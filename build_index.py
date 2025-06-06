from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_data():
    topics = [
        "iPhone",
        "iPhone 16",
        "iPhone 16 Pro",
        "iPhone 15 Pro",
        "iPhone 15",
        "iPhone 14",
        "iPhone 14 Pro",
        "iPhone 13",
        "iPhone 13 Pro",
        "iPhone 12",
    ]

    loader = WikipediaLoader(query=topics)
    docs = loader.load()
    return docs

def create_knowledgebase():
    docs = load_data()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)

     # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name='all-MiniLM-L6-v2',
        encode_kwargs={'normalize_embeddings': True}
    )

    # Create and save FAISS vector store
    vector_store = FAISS.from_documents(split_docs, embeddings)
    vector_store.save_local("vectorstore.db")
    print("Knowledge base saved to 'vectorstore.db'")

if __name__ == "__main__":
    create_knowledgebase()