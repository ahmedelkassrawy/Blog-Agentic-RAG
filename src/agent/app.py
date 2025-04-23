from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END, MessagesState, START
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Annotated
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_community.document_loaders.firecrawl import FireCrawlLoader
from langchain_community.tools import ElevenLabsText2SpeechTool
import os
from typing import List, TypedDict
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import NewsURLLoader
from langchain.chains.summarize import load_summarize_chain
from langgraph.graph import StateGraph, START, END

# API Keys
os.environ["ELEVENLABS_API_KEY"] = "sk_73447df115bbeaa2849987b6f54293a8e2529677aed8f558"
os.environ["FIRECRAWL_API_KEY"] = "fc-27dd402673704e9c8f3d982668efcbdf"


llm = ChatGroq(
    api_key = "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",
    model = "llama-3.3-70b-versatile"
)

# Define State structure
class TrendState(TypedDict):
    topic: str
    articles: List[str]
    summaries: str
    analysis: str


# News Collector Node
def news_collector_node(state: TrendState) -> TrendState:
    search_tool = DuckDuckGoSearchRun()
    query = f"{state['topic']} site:news"
    search_results = search_tool.run(query)
    # naive extraction of URLs from results (split by newlines)
    articles = [line for line in search_results.split('\n') if line.startswith("http")]
    return {"topic": state["topic"], "articles": articles}

# Summary Writer Node
def summary_writer_node(state: TrendState) -> TrendState:
    loader = NewsURLLoader(urls=state["articles"])
    docs = loader.load()
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summaries = chain.run(docs)
    return {"topic": state["topic"], "articles": state["articles"], "summaries": summaries}

# Trend Analyzer Node
def trend_analyzer_node(state: TrendState) -> TrendState:
    prompt = f"""
    Analyze the following news summaries and identify:
    - Emerging industry trends
    - Gaps or problems that startups can solve
    - Any interesting startup examples or innovations

    Summaries:
    {state['summaries']}
    """
    result = llm.invoke(prompt)
    return {
        "topic": state["topic"],
        "articles": state["articles"],
        "summaries": state["summaries"],
        "analysis": result.content
    }

# Build the graph
builder = StateGraph(TrendState)
builder.add_node("NewsCollector", news_collector_node)
builder.add_node("SummaryWriter", summary_writer_node)
builder.add_node("TrendAnalyzer", trend_analyzer_node)

builder.set_entry_point("NewsCollector")
builder.add_edge("NewsCollector", "SummaryWriter")
builder.add_edge("SummaryWriter", "TrendAnalyzer")
builder.add_edge("TrendAnalyzer", END)

graph = builder.compile()


topic = input("Enter the area of interest for your Startup: ")
input_state = {"topic": topic}
result = graph.invoke(input_state)
print("\n=== Trend Analysis and Startup Opportunities ===\n")
print(result["analysis"])
