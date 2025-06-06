import streamlit as st
from query_groq import ask_query, load_retriever
import os
from dotenv import load_dotenv

st.title("Ask anything about the iPhones")

# Load environment variables early
# load_dotenv()

# Cache the retriever to avoid reloading each time
@st.cache_resource
def get_cached_retriever():
    return load_retriever()

retriever = get_cached_retriever()

# query_model = "gpt2"
groq_model = "llama-3.1-8b-instant"
groq_token = st.secrets["groq_api_key"]

#chat history list
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
#button to clear chat history
if st.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.success("Chat history cleared!")
# Display Chat history
with st.expander(" Chat History", expanded=True):
    if st.session_state.chat_history:
        for i, (q, a) in enumerate(st.session_state.chat_history):
            st.markdown(f"**User:** {q}")
            st.markdown(f"**iBot:** {a}")
            st.markdown("---")
    else:
        st.markdown("_No previous conversation yet._")

query = st.text_area("Your question:")
show_sources = st.checkbox("Show sources")


if st.button("Ask"):
    with st.spinner("Generating answer..."):
        try:
            
            answer, sources = ask_query(groq_model, query, groq_token, retriever,st.session_state.chat_history)
            st.session_state.chat_history.append((query, answer))
            st.markdown("### Answer")
            st.write(answer)
            
            if show_sources and sources:
                st.markdown("### 📚 Retrieved Documents")
                for i, doc in enumerate(sources):
                    st.markdown(f"**Source {i+1}:** {doc.page_content[:500]}...")  # limit output length
        except Exception as e:
            st.error(f"Error: {e}")
