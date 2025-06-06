from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name='all-MiniLM-L6-v2',
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_store = FAISS.load_local("vectorstore.db", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    return retriever

def ask_query(groq_model, query, groq_token, retriever):
    
    llm = ChatGroq(
        model=groq_model,
        temperature=0.7,
        max_retries=2,
        max_tokens = 512,
        api_key= groq_token,
    )

    # You can keep your original prompt style for text-generation:
    template = """You are a helpful and expert assistant specialized in Apple iPhones. Use only the context provided below to answer the question. Do not use outside knowledge or make assumptions.

        Context:
        {context}

        Question:
        {input}

        Answer:"""



    prompt = ChatPromptTemplate.from_template(template)
    # retriever = load_retriever()
    doc_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, doc_chain)

    response = chain.invoke({"input": query}, return_only_outputs=False)
    answer = response['answer']
    sources = response.get("context", [])

    return answer, sources
