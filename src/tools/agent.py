# agent.py
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain import hub

from pydantic import BaseModel, Field
from typing import Literal
from functools import partial

# Define State
class AgentState(MessagesState): pass

# Agent Nodes
def agent(state: AgentState, tools):
    llm = get_llm()
    response = llm.bind_tools(tools).invoke(state["messages"])
    return {"messages": [response]}

def rewrite(state: AgentState):
    llm = get_llm()
    question = state["messages"][0].content
    msg = [HumanMessage(content=f"""
        Look at the input and try to reason about the underlying semantic intent / meaning.
        Here is the initial question:\n ------- \n{question}\n ------- \n
        Formulate an improved question:
    """)]
    response = llm.invoke(msg)
    return {"messages": [response]}

def generate(state: AgentState):
    llm = get_llm()

    question = state["messages"][0].content
    docs = state["messages"][-1].content

    prompt = hub.pull("rlm/rag-prompt")

    rag_chain = prompt | llm | StrOutputParser()

    response = rag_chain.invoke({"context": docs, 
                                 "question": question})
    return {"messages": [response]}

def grade_docs(state: AgentState) -> Literal["generate", "rewrite"]:
    class Grade(BaseModel):
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    llm = get_llm()
    llm_with_tool = llm.with_structured_output(Grade)

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question.
        \nHere is the retrieved document: \n\n{context}\n\n
        Here is the user question: {question}
        \nIf the document contains keyword(s) or semantic meaning related to the user question,
        grade it as relevant. \nGive a binary score 'yes' or 'no'.""",
        input_variables=["context", "question"],
    )

    chain = prompt | llm_with_tool

    question = state["messages"][0].content
    docs = state["messages"][-1].content

    response = chain.invoke({"context": docs, 
                             "question": question})
    
    return "generate" if response.binary_score == "yes" else "rewrite"

# Utility
def get_llm():
    import os
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0,
    )

def create_graph(embedding, retriever_tool):
    tools = [retriever_tool]
    builder = StateGraph(AgentState)
    builder.add_node("retrieve", ToolNode(tools))
    builder.add_node("agent", partial(agent, tools=tools))
    builder.add_node("rewrite", rewrite)
    builder.add_node("generate", generate)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition, {
        "tools": "retrieve",
        "continue": "rewrite"
    })
    builder.add_conditional_edges("retrieve", grade_docs, {
        "generate": "generate",
        "rewrite": "rewrite"
    })
    builder.add_edge("generate", END)
    builder.add_edge("rewrite", "agent")

    return builder.compile()

def setup_vectorstore(url):
    import os

    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=100, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
    )

    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="qdrant_db",
        url=os.environ.get("QDRANT_URL"),
        api_key=os.environ.get("QDRANT_API_KEY"),
        force_recreate=True
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    retriever_tool = create_retriever_tool(
        retriever,
        "retriever_blog_posts",
        "Search and return information about blog posts on LLMs, LLM agents, prompt engineering, and adversarial attacks on LLMs.",
    )

    return embeddings, retriever_tool
