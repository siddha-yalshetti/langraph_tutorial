import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph_backend import chatbot

st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖",
    layout="centered"
)
st.title("🤖 LangGraph AI Chatbot")
st.caption(
    "Chatbot powered by LangGraph + Groq"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "user_1"

with st.sidebar:

    st.header("⚙️ Settings")

    st.write(
        "Thread ID:"
    )

    st.code(
        st.session_state.thread_id
    )

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        # Create new thread
        st.session_state.thread_id = (
            "user_"
            + str(
                id(st.session_state)
            )
        )

        st.rerun()

for message in st.session_state.messages:

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
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )


    # LangGraph thread configuration
    config = {
        "configurable": {
            "thread_id":
                st.session_state.thread_id
        }
    }


    # Generate response
    with st.chat_message("assistant"):

        with st.spinner(
            "Thinking..."
        ):

            try:

                response = chatbot.invoke(
                    {
                        "messages": [
                            HumanMessage(
                                content=user_input
                            )
                        ]
                    },
                    config=config
                )


                # Get AI response
                ai_response = (
                    response["messages"][-1].content
                )


                # Display response
                st.write(
                    ai_response
                )


                # Save AI response
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": ai_response
                    }
                )


            except Exception as e:

                st.error(
                    f"Error: {e}"
                )