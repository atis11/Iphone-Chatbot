from query_groq import load_retriever, ask_query
import os
retriever = load_retriever()
query = "What is the difference between iPhone 14 and iPhone 15?"
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
groq_api_token = os.getenv("groq_api_key")
query_model="meta-llama/Llama-3.1-8B-Instruct"
answer, sources = ask_query(query_model,query,groq_api_token,retriever
)

print("Answer:", answer)
# for doc in sources:
#     print("\nSource:", doc.metadata.get("source", "N/A"))
#     print(doc.page_content[:300])
