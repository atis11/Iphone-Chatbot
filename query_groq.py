from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name='all-MiniLM-L6-v2',
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_store = FAISS.load_local("vectorstore.db", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return retriever

def ask_query(groq_model, query, groq_token, retriever, chat_history=[]):
    llm = ChatGroq(
        model=groq_model,
        temperature=0.7,
        max_retries=2,
        max_tokens=512,
        api_key=groq_token,
    )

    template = """
You are a helpful assistant answering questions about iPhones. Answer the user's question based on the given conversation history and the given context.

Keep track of the device user asked recently and answer according to it if name of the device is not mentioned properly.
For example:
user: which is the latest iphone?
answer: iphone 16 pro max.

user: How many camera does it have?
answer: iphone 16 pro max has 3 camera.

user: Is it better than iphone 15 pro in terms of camera?
answer: Yes iphone 16 pro max has better camera than iphone 15 pro.

Context:
{context}

Conversation history:
{chat_history}

User question:
{question}

Answer:
"""
    prompt = ChatPromptTemplate.from_template(template)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
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
