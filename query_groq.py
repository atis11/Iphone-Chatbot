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
You are a helpful and knowledgeable assistant that answers questions about iPhones based on the provided context and conversation history.

Your job is to:
1. Answer the user's question as accurately as possible using the provided context.
2. Track the most recently mentioned iPhone model from the user's side in the conversation history if the current question refers to "this phone", "it", etc.
3. If the user explicitly asks about a different iPhone model (e.g., "Tell me about iPhone 14 Pro Max"), prioritize answering that directly using the context, even if it doesn't match prior history.

Examples:

Conversation:
User: Which is the latest iPhone?
Assistant: The iPhone 16 Pro Max is the latest.

User: How many cameras does it have?
Assistant: The iPhone 16 Pro Max has three rear cameras and a front-facing camera.

User: Tell me about the specs of iPhone 14 Pro Max.
Assistant: [Answer should use context and focus on iPhone 14 Pro Max.]

If the user's current question does not refer clearly to a specific device, assume they are asking about the one most recently discussed in the conversation history.

---

Context (information from Apple sources):
{context}

Conversation History:
{chat_history}

User Question:
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
