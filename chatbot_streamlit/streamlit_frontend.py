import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph_backend import (
    chatbot,
    create_thread,
    get_thread_messages,
    list_threads,
    update_thread_title,
)

st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖",
    layout="centered"
)
st.title("🤖 LangGraph AI Chatbot")
st.caption(
    "Chatbot powered by LangGraph + Groq"
)

threads = list_threads()
if not threads:
    create_thread()
    threads = list_threads()

thread_ids = [thread["thread_id"] for thread in threads]
if "active_chat_id" not in st.session_state or st.session_state.active_chat_id not in thread_ids:
    st.session_state.active_chat_id = thread_ids[0]


st.session_state.thread_id = st.session_state.active_chat_id


def create_new_chat():
    st.session_state.active_chat_id = create_thread()
    st.session_state.thread_id = st.session_state.active_chat_id


def select_chat(thread_id):
    st.session_state.active_chat_id = thread_id
    st.session_state.thread_id = thread_id


st.session_state.thread_id = st.session_state.active_chat_id
active_thread = next(
    thread for thread in threads
    if thread["thread_id"] == st.session_state.active_chat_id
)
active_messages = get_thread_messages(st.session_state.active_chat_id)

with st.sidebar:

    st.title("LangGraph Chatbot")

    if st.button(
        "➕ Start Chat",
        use_container_width=True,
        type="primary"
    ):
        create_new_chat()
        st.rerun()

    st.subheader("My Conversation")
    st.caption("Current thread ID")
    st.code(st.session_state.thread_id)

    for thread in threads:
        thread_id = thread["thread_id"]
        button_type = "primary" if thread_id == st.session_state.active_chat_id else "secondary"
        st.caption(thread["title"])
        if st.button(
            thread_id,
            key=f"thread_{thread_id}",
            use_container_width=True,
            type=button_type
        ):
            select_chat(thread_id)
            st.rerun()

    st.caption("Conversations are saved while this app session is running.")

for message in active_messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )

user_input = st.chat_input(
    "Type your message..."
)

if user_input:

    # Display user message
    with st.chat_message("user"):

        st.write(user_input)


    # Save user message
    if active_thread["title"] == "New conversation":
        title = user_input[:32]
        if len(user_input) > 32:
            title += "..."
        update_thread_title(st.session_state.thread_id, title)


    # LangGraph thread configuration
    # config = {
    #     "configurable": {
    #                 "thread_id": st.session_state.thread_id
    #     }
    # }

    config={
        'configurable':{"thread_id":st.session_state["thread_id"]},
        "metadata":{
            "thread_id":st.session_state["thread_id"]
        },
        "run_name":"chat_turn",
    }


    # Generate response
    with st.chat_message("assistant"):

        try:

            def stream_response():
                for message_chunk, _metadata in chatbot.stream(
                    {
                        "messages": [
                            HumanMessage(
                                content=user_input
                            )
                        ]
                    },
                    config=config,
                    stream_mode="messages"
                ):
                    if message_chunk.content:
                        yield message_chunk.content

            # Render each token as it arrives.
            ai_response = st.write_stream(stream_response())

        except Exception as e:

            st.error(
                f"Error: {e}"
            )