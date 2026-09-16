from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os

load_dotenv()

# 1. Define the state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# 2. Create the LLM
model = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.environ.get("GROK_API_KEY")
)


# 3. Create the checkpointer
checkpointer = InMemorySaver()


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