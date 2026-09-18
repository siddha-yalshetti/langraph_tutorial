from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4

load_dotenv()

# 1. Define the state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# 2. Create the LLM
model = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.environ.get("GROK_API_KEY")
)


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
    response = model.invoke(state["messages"])
    return {"messages": [response]}


# 6. Add node
graph.add_node("chat_node", chat_node)


# 7. Define edges
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)


# 8. Compile with checkpointer
chatbot = graph.compile(
    checkpointer=checkpointer
)

