from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
import os
import sqlite3
import json
import re
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from datetime import datetime, timezone
from uuid import uuid4

load_dotenv()

# 1. Define the state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@tool
def get_stock_price(symbol: str) -> str:
    """Get the latest available stock quote for a ticker symbol, such as AAPL or MSFT."""
    ticker = symbol.strip().upper()
    if not re.fullmatch(r"[A-Z0-9.=-]{1,12}", ticker):
        return "Invalid ticker symbol. Use a symbol such as AAPL, MSFT, or TSLA."

    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return "Stock data is not configured. Add ALPHA_VANTAGE_API_KEY to the .env file."

    query = urlencode({
        "function": "GLOBAL_QUOTE",
        "symbol": ticker,
        "apikey": api_key,
    })
    url = f"https://www.alphavantage.co/query?{query}"
    request = Request(url, headers={"User-Agent": "LangGraph-Chatbot/1.0"})

    try:
        with urlopen(request, timeout=10) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, ValueError) as error:
        return f"Could not retrieve a quote for {ticker}: {error}"

    if "Note" in payload:
        return f"Stock data provider limit reached: {payload['Note']}"
    if "Information" in payload:
        return f"Stock data provider response: {payload['Information']}"

    quote = payload.get("Global Quote", {})
    price = quote.get("05. price")
    if not price:
        return f"No stock data was found for {ticker}. Check the ticker symbol."

    change = quote.get("09. change", "unknown")
    change_percent = quote.get("10. change percent", "unknown")
    timestamp = quote.get("07. latest trading day", "the latest available market data")
    return (
        f"{ticker}: ${float(price):.2f}, change {change} ({change_percent}), "
        f"latest trading day {timestamp}. Market data may be delayed."
    )


@tool
def get_latest_news(topic: str) -> str:
    """Get the latest news headlines about a topic, company, person, or event."""
    search_topic = topic.strip()
    if not search_topic:
        return "Please provide a topic, company, person, or event to search for."

    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        return "News data is not configured. Add NEWS_API_KEY to the .env file."

    query = urlencode({
        "q": search_topic,
        "sortBy": "publishedAt",
        "pageSize": 5,
        "language": "en",
        "apiKey": api_key,
    })
    request = Request(
        f"https://newsapi.org/v2/everything?{query}",
        headers={"User-Agent": "LangGraph-Chatbot/1.0"},
    )

    try:
        with urlopen(request, timeout=10) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, ValueError) as error:
        return f"Could not retrieve news for {search_topic}: {error}"

    if payload.get("status") != "ok":
        error_message = payload.get("message", "Unknown news provider error")
        return f"News provider error: {error_message}"

    articles = payload.get("articles", [])
    if not articles:
        return f"No recent news was found for {search_topic}."

    headlines = []
    for article in articles:
        title = article.get("title", "Untitled")
        source = article.get("source", {}).get("name", "Unknown source")
        published_at = article.get("publishedAt", "unknown time")
        url = article.get("url", "")
        headlines.append(f"- {title} | {source} | {published_at} | {url}")

    return f"Latest news for {search_topic}:\n" + "\n".join(headlines)


# 2. Create the LLM
model = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.environ.get("GROK_API_KEY")
).bind_tools([get_stock_price, get_latest_news])


# 3. Create the persistent SQLite checkpointer and thread catalog
database_path = os.path.join(os.path.dirname(__file__), "chatbot.db")
database_connection = sqlite3.connect(database_path, check_same_thread=False)
checkpointer = SqliteSaver(database_connection)
checkpointer.setup()

database_connection.execute(
    """
    CREATE TABLE IF NOT EXISTS threads (
        thread_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """
)
database_connection.commit()


def create_thread(title="New conversation"):
    thread_id = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()
    database_connection.execute(
        "INSERT INTO threads (thread_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (thread_id, title, now, now)
    )
    database_connection.commit()
    return thread_id


def list_threads():
    rows = database_connection.execute(
        "SELECT thread_id, title FROM threads ORDER BY updated_at DESC"
    ).fetchall()
    return [{"thread_id": row[0], "title": row[1]} for row in rows]


def update_thread_title(thread_id, title):
    now = datetime.now(timezone.utc).isoformat()
    database_connection.execute(
        "UPDATE threads SET title = ?, updated_at = ? WHERE thread_id = ?",
        (title, now, thread_id)
    )
    database_connection.commit()


def get_thread_messages(thread_id):
    state = chatbot.get_state({"configurable": {"thread_id": thread_id}})
    messages = [] if not state or not state.values else state.values.get("messages", [])
    return [
        {
            "role": "user" if message.type == "human" else "assistant",
            "content": message.content
        }
        for message in messages
        if message.type in {"human", "ai"} and isinstance(message.content, str)
    ]


# 4. Create the graph
graph = StateGraph(ChatState)


# 5. Create chat node
def chat_node(state: ChatState):
    system_message = (
        "You are a helpful assistant. For current stock prices, always use the "
        "get_stock_price tool rather than guessing. Ask for a ticker symbol if "
        "the company is ambiguous. Clearly state that market data may be delayed. "
        "For current news or headlines, always use the get_latest_news tool "
        "rather than relying on memory. Include the source and link when reporting news."
    )
    response = model.invoke(
        [{"role": "system", "content": system_message}] + state["messages"]
    )
    return {"messages": [response]}


# 6. Add node
graph.add_node("chat_node", chat_node)
graph.add_node("tools", ToolNode([get_stock_price, get_latest_news]))


# 7. Define edges
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")


# 8. Compile with checkpointer
chatbot = graph.compile(
    checkpointer=checkpointer
)

