import uuid
import streamlit as st
from langchain_core.messages import HumanMessage
from app.graph import build_graph

st.set_page_config(
    page_title="MediBuddy Weather Safety Advisor",
    page_icon="🌤️",
    layout="wide",
)

# Custom styling for visual clarity
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #e9ecef;
        text-align: center;
    }
    .badge-critical { background-color: #dc3545; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-high { background-color: #fd7e14; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-medium { background-color: #ffc107; color: black; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-low { background-color: #28a745; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_graph():
    if "compiled_graph" not in st.session_state:
        st.session_state.compiled_graph = build_graph()
    return st.session_state.compiled_graph


def initialize_session():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []


def reset_chat():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    if "compiled_graph" in st.session_state:
        del st.session_state["compiled_graph"]


def run_query(query: str) -> dict:
    graph = get_graph()
    result = graph.invoke(
        {
            "user_query": query,
            "messages": [HumanMessage(content=query)],
        },
        config={"configurable": {"thread_id": st.session_state.thread_id}},
    )
    return result


initialize_session()

# Sidebar Controls
with st.sidebar:
    st.title("⚙️ Session Panel")
    st.caption("Active Session Details")
    st.code(f"ID: {st.session_state.thread_id[:8]}...", language="text")

    st.markdown("---")
    st.markdown("### 💡 Try Asking:")
    example_prompts = [
        "Is it safe to cycle in Bhopal right now?",
        "What about this evening?",
        "Can I do the same in Lucknow tomorrow?",
        "Should I take my kids to the park in Delhi?",
        "Would today be good for a picnic in Pune?",
    ]
    for prompt in example_prompts:
        if st.button(prompt, key=f"btn_{prompt[:12]}", use_container_width=True):
            st.session_state.suggested_prompt = prompt

    st.markdown("---")
    if st.button("🗑️ Reset Chat", use_container_width=True):
        reset_chat()
        st.rerun()

# Main Header
st.title("🌤️ MediBuddy Weather-Advisory Bot")
st.caption("Deterministic outdoor activity safety recommendations grounded in live Open-Meteo data.")

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Render stored weather badges if available in previous turn
        if "meta" in message and message["meta"]:
            weather = message["meta"].get("weather")
            sop = message["meta"].get("sop")
            if weather and sop:
                with st.expander("🔍 Policy & Weather Traceability", expanded=False):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Temperature", f"{weather.temperature}°C")
                    col2.metric("Wind Speed", f"{weather.wind_speed} km/h")
                    col3.metric("Rain Prob.", f"{weather.precipitation_probability}%")
                    col4.metric("UV Index", f"{weather.uv_index}")
                    st.markdown(f"**Applied Policy:** `{sop.id}` | **Severity:** `{sop.severity.upper()}`")

# Input Handling (supporting quick prompt buttons)
suggested_query = st.session_state.pop("suggested_prompt", None)
query = st.chat_input("Ask a safety question (e.g., Can I bike to work in Bhopal?)") or suggested_query

if query:
    # Append & display user prompt
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Process and stream assistant response
    with st.chat_message("assistant"):
        with st.spinner("Checking live weather conditions and evaluating safety policies..."):
            meta = {}
            try:
                result = run_query(query)
                response_text = result.get(
                    "final_response",
                    "I don't have guidance for that situation."
                )

                weather = result.get("weather")
                sop = result.get("selected_sop")
                meta = {"weather": weather, "sop": sop}

                st.markdown(response_text)

                # Show real-time weather cards & policy match
                if weather or sop:
                    with st.expander("🔍 Policy & Weather Traceability", expanded=True):
                        if weather:
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Temperature", f"{weather.temperature}°C")
                            col2.metric("Wind Speed", f"{weather.wind_speed} km/h")
                            col3.metric("Rain Prob.", f"{weather.precipitation_probability}%")
                            col4.metric("UV Index", f"{weather.uv_index}")
                            st.caption(f"📍 Location: **{weather.location}** | Recorded Time: `{weather.timestamp}`")

                        if sop:
                            st.markdown(
                                f"**Rule Reference:** `{sop.id}` — *{sop.category.replace('_', ' ').title()}* | "
                                f"**Severity:** <span class='badge-{sop.severity}'>{sop.severity.upper()}</span>",
                                unsafe_allow_html=True,
                            )
                            st.info(f"**Standard Operating Procedure:** {sop.advice}")

            except Exception as exc:
                response_text = (
                    "An unexpected error occurred while communicating with the weather services. Please try again."
                )
                st.error(f"Execution Exception: {type(exc).__name__}: {exc}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "meta": meta,
    })