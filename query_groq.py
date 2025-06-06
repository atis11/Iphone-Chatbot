from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
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

def ask_query(groq_model, query, groq_token, retriever, chat_history=[]):
    llm = ChatGroq(
        model=groq_model,
        temperature=0.7,
        max_retries=2,
        max_tokens=512,
        api_key=groq_token,
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )

    result = qa_chain.invoke({
        "question": query,
        "chat_history": chat_history
    })

    answer = result["answer"]
    sources = result.get("source_documents", [])

    return answer, sources
