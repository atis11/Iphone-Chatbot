import streamlit as st
from query_groq import ask_query, load_retriever
import os
from dotenv import load_dotenv

st.title("Ask anything about the iPhones")

# Load environment variables early
load_dotenv()

# Cache the retriever to avoid reloading each time
@st.cache_resource
def get_cached_retriever():
    return load_retriever()

retriever = get_cached_retriever()

# query_model = "gpt2"
groq_model = "llama-3.1-8b-instant"
groq_token = os.getenv("groq_api_key")

query = st.text_area("Your question:")
show_sources = st.checkbox("Show sources")

if st.button("Ask"):
    with st.spinner("Generating answer..."):
        try:
            answer, sources = ask_query(groq_model, query, groq_token, retriever)
            st.markdown("### Answer")
            st.write(answer)
            
            if show_sources and sources:
                st.markdown("### 📚 Retrieved Documents")
                for i, doc in enumerate(sources):
                    st.markdown(f"**Source {i+1}:** {doc.page_content[:500]}...")  # limit output length
        except Exception as e:
            st.error(f"Error: {e}")
