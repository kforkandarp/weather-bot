import uuid

import streamlit as st

from app.graph import build_graph
from langchain_core.messages import HumanMessage


st.set_page_config(
    page_title="MediBuddy Weather Decision Bot",
    page_icon="🌤️",
    layout="centered",
)


@st.cache_resource
def get_graph():
    return build_graph()


def initialize_session():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    if "messages" not in st.session_state:
        st.session_state.messages = []


def reset_chat():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []


def run_query(query: str) -> str:
    graph = get_graph()

    result = graph.invoke(
        {
            "user_query": query,
            "messages": [HumanMessage(content=query)],
        },
        config={
            "configurable": {
                "thread_id": st.session_state.thread_id
            }
        },
    )

    return result.get(
        "final_response",
        "I couldn't generate a response."
    )


initialize_session()


st.title("🌤️ MediBuddy Weather Decision Bot")

st.caption(
    "Live weather + policy-controlled safety advice powered by LangGraph"
)


with st.sidebar:
    st.header("About")

    st.write(
        """
        Ask whether an outdoor activity is suitable based on:

        - Live Open-Meteo weather
        - Location resolution
        - Semantic SOP retrieval
        - Deterministic policy evaluation
        - Policy-based response generation
        """
    )

    st.divider()

    st.write("**Session memory:** active")

    if st.button("🗑️ Clear conversation", use_container_width=True):
        reset_chat()
        st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


query = st.chat_input(
    "Ask something like: Is it safe to cycle in Bhopal today?"
)


if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        with st.spinner("Checking weather and applicable policies..."):

            try:
                response = run_query(query)

            except Exception as exc:
                response = (
                    "Something went wrong while processing the request. "
                    "Please try again."
                )

                st.error(
                    f"Application error: {type(exc).__name__}: {exc}"
                )

        st.markdown(response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )