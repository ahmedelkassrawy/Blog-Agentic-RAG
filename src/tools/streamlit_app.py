# main.py
import os
import streamlit as st
from langchain_core.messages import HumanMessage
from agent import create_graph, setup_vectorstore, get_llm

st.set_page_config(
    page_title="AI Blog Search",
    page_icon="🧠",
)
st.header(":blue[Agentic RAG with LangGraph:] :green[AI Blog Search]")


with st.sidebar:
    st.subheader("API Keys")
    qdrant_url = st.text_input("Qdrant URL", type="password")
    qdrant_api_key = st.text_input("Qdrant API Key", type="password")
    google_api_key = st.text_input("Google API Key", type="password")

    if st.button("Done"):
        st.session_state.qdrant_url = qdrant_url
        st.session_state.qdrant_api_key = qdrant_api_key
        st.session_state.google_api_key = google_api_key
        st.success("API keys saved successfully!")

# Set env vars
os.environ["QDRANT_URL"] = st.session_state.get("qdrant_url", "")
os.environ["QDRANT_API_KEY"] = st.session_state.get("qdrant_api_key", "")
os.environ["GOOGLE_API_KEY"] = st.session_state.get("google_api_key", "")

# Initialize session state for embeddings and retriever_tool
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "retriever_tool" not in st.session_state:
    st.session_state.retriever_tool = None

url = st.text_input("Enter the URL of the blog post to analyze:")

if st.button("Load Blog Post") and url:
    with st.spinner():
        embeddings, retriever_tool = setup_vectorstore(url)

        if embeddings and retriever_tool:
            st.session_state.embeddings = embeddings
            st.session_state.retriever_tool = retriever_tool
            st.success("Blog post loaded successfully!")
        else:
            st.error("Failed to load the blog post. Please check the URL and try again.")

# Ensure graph creation only happens if embeddings and retriever_tool are valid
if st.session_state.embeddings and st.session_state.retriever_tool:
    graph = create_graph(st.session_state.embeddings, st.session_state.retriever_tool)

    # UI Query Input
    query = st.text_input("Ask something about LLM agents...")

    if st.button("Run Query") and query:
        st.info("Running the agent pipeline...")
        result = graph.invoke({"messages": [HumanMessage(content=query)]})
        st.success("Response:")
        result["messages"][-1].content
    else:
        st.warning("Please enter a query to run the agent pipeline.")
else:
    st.warning("Please load a blog post before running queries.")

st.markdown("---")
st.write("Built with :blue-background[LangChain] | :blue-background[LangGraph]")