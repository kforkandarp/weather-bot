from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from app.evaluator import evaluate_sops, select_sop
from app.intent import extract_intent
from app.llm import get_llm
from app.location import LocationResult, resolve_location
from app.sop_loader import load_sops
from app.state import AgentState, SOP, UserIntent, WeatherState
from app.weather import fetch_weather


# ---------------------------------------------------------
# NODES
# ---------------------------------------------------------

def extract_intent_node(state: AgentState) -> AgentState:
    intent = extract_intent(
        user_query=state["user_query"],
        previous_intent=state.get("intent"),
    )
    return {
        "intent": intent,
        "outcome": None,
        "final_response": None,
    }


def check_location_node(state: AgentState) -> AgentState:
    intent = state.get("intent")
    if intent is None or not intent.location:
        return {"outcome": "MISSING_LOCATION"}
    return {"outcome": None}


def missing_location_node(state: AgentState) -> AgentState:
    return {
        "final_response": "What city or location should I check the weather for?"
    }


def resolve_location_node(state: AgentState) -> AgentState:
    intent = state["intent"]
    if intent is None or not intent.location:
        return {"outcome": "MISSING_LOCATION"}

    location = resolve_location(intent.location)
    if location is None:
        return {"outcome": "LOCATION_RESOLUTION_FAILED"}

    return {
        "resolved_location": location.name,
        "latitude": location.latitude,
        "longitude": location.longitude,
        "outcome": None,
    }


def location_failure_node(state: AgentState) -> AgentState:
    return {
        "final_response": (
            "I couldn't resolve that location, so I can't safely check live weather for it."
        )
    }


def fetch_weather_node(state: AgentState) -> AgentState:
    intent = state.get("intent")
    resolved_location = state.get("resolved_location")
    latitude = state.get("latitude")
    longitude = state.get("longitude")

    if (
        not resolved_location
        or latitude is None
        or longitude is None
        or intent is None
    ):
        return {"outcome": "WEATHER_FETCH_FAILED"}

    location = LocationResult(
        name=resolved_location,
        latitude=latitude,
        longitude=longitude,
    )

    weather = fetch_weather(
        location=location,
        time_expression=intent.time_expression,
        requested_hour=intent.hour,
    )

    if weather is None:
        return {"outcome": "WEATHER_FETCH_FAILED"}

    return {
        "weather": weather,
        "outcome": None,
    }


def weather_failure_node(state: AgentState) -> AgentState:
    return {
        "final_response": (
            "I couldn't retrieve live weather for that location, so I can't give a policy-based recommendation."
        )
    }


def retrieve_sops_node(state: AgentState) -> AgentState:
    # Always load fresh from disk to support live SOP updates
    return {
        "candidate_sops": load_sops()
    }


def evaluate_sops_node(state: AgentState) -> AgentState:
    intent = state.get("intent")
    weather = state.get("weather")
    candidate_sops = state.get("candidate_sops", [])

    if intent is None or weather is None:
        return {
            "applicable_sops": [],
            "outcome": "NO_APPLICABLE_SOP",
        }

    applicable_sops = evaluate_sops(
        sops=candidate_sops,
        intent=intent,
        weather=weather,
    )

    return {
        "applicable_sops": applicable_sops,
    }


def no_applicable_sop_node(state: AgentState) -> AgentState:
    return {
        "outcome": "NO_APPLICABLE_SOP",
        "final_response": "I don't have guidance for that situation.",
    }


def resolve_sop_node(state: AgentState) -> AgentState:
    applicable = state.get("applicable_sops", [])
    selected_sop = select_sop(applicable)

    if selected_sop is None:
        return {
            "selected_sop": None,
            "outcome": "NO_APPLICABLE_SOP",
        }

    return {
        "selected_sop": selected_sop,
        "outcome": "SUCCESS",
    }


def final_response_node(state: AgentState) -> AgentState:
    weather = state["weather"]
    sop = state["selected_sop"]
    llm = get_llm()

    prompt = f"""
You are the final response writer for a weather decision system.
The policy decision has already been made deterministically.
Your ONLY job is to explain the selected policy decision clearly.

Do not:
- select another SOP
- change the decision
- invent weather facts
- modify weather numbers
- add advice not present in the SOP
- contradict the SOP

User question:
{state["user_query"]}

Live weather from Open-Meteo:
Location: {weather.location}
Time: {weather.timestamp}
Temperature: {weather.temperature} {weather.temperature_unit}
Wind speed: {weather.wind_speed} {weather.wind_speed_unit}
Precipitation: {weather.precipitation} {weather.precipitation_unit}
Precipitation probability: {weather.precipitation_probability}%
UV index: {weather.uv_index}

Selected policy:
SOP ID: {sop.id}
Severity: {sop.severity}
Advice: {sop.advice}

Give a concise answer.
Use the actual weather values above.
Explain why the selected SOP applies.
Explicitly cite the SOP ID (e.g. {sop.id}).
Follow the SOP advice exactly in substance.
"""
    response = llm.invoke(prompt)
    return {
        "final_response": response.content
    }


# ---------------------------------------------------------
# ROUTING
# ---------------------------------------------------------

def route_location_check(state: AgentState) -> str:
    if state.get("outcome") == "MISSING_LOCATION":
        return "missing_location"
    return "resolve_location"


def route_location_resolution(state: AgentState) -> str:
    if state.get("outcome") == "LOCATION_RESOLUTION_FAILED":
        return "location_failure"
    return "fetch_weather"


def route_weather(state: AgentState) -> str:
    if state.get("outcome") == "WEATHER_FETCH_FAILED":
        return "weather_failure"
    return "retrieve_sops"


def route_evaluation(state: AgentState) -> str:
    if not state.get("applicable_sops"):
        return "no_applicable_sop"
    return "resolve_sop"


def route_sop_resolution(state: AgentState) -> str:
    if state.get("outcome") == "NO_APPLICABLE_SOP" or state.get("selected_sop") is None:
        return "no_applicable_sop"
    return "final_response"


# ---------------------------------------------------------
# GRAPH
# ---------------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("extract_intent", extract_intent_node)
    graph.add_node("check_location", check_location_node)
    graph.add_node("missing_location", missing_location_node)
    graph.add_node("resolve_location", resolve_location_node)
    graph.add_node("location_failure", location_failure_node)
    graph.add_node("fetch_weather", fetch_weather_node)
    graph.add_node("weather_failure", weather_failure_node)
    graph.add_node("retrieve_sops", retrieve_sops_node)
    graph.add_node("evaluate_sops", evaluate_sops_node)
    graph.add_node("no_applicable_sop", no_applicable_sop_node)
    graph.add_node("resolve_sop", resolve_sop_node)
    graph.add_node("final_response", final_response_node)

    graph.add_edge(START, "extract_intent")
    graph.add_edge("extract_intent", "check_location")

    graph.add_conditional_edges(
        "check_location",
        route_location_check,
        {
            "missing_location": "missing_location",
            "resolve_location": "resolve_location",
        },
    )
    graph.add_edge("missing_location", END)

    graph.add_conditional_edges(
        "resolve_location",
        route_location_resolution,
        {
            "location_failure": "location_failure",
            "fetch_weather": "fetch_weather",
        },
    )
    graph.add_edge("location_failure", END)

    graph.add_conditional_edges(
        "fetch_weather",
        route_weather,
        {
            "weather_failure": "weather_failure",
            "retrieve_sops": "retrieve_sops",
        },
    )
    graph.add_edge("weather_failure", END)

    graph.add_edge("retrieve_sops", "evaluate_sops")

    graph.add_conditional_edges(
        "evaluate_sops",
        route_evaluation,
        {
            "no_applicable_sop": "no_applicable_sop",
            "resolve_sop": "resolve_sop",
        },
    )
    graph.add_edge("no_applicable_sop", END)

    graph.add_conditional_edges(
        "resolve_sop",
        route_sop_resolution,
        {
            "no_applicable_sop": "no_applicable_sop",
            "final_response": "final_response",
        },
    )
    graph.add_edge("final_response", END)

    checkpointer = MemorySaver()
    checkpointer.serde = JsonPlusSerializer(
        allowed_msgpack_modules=[UserIntent, WeatherState, SOP]
    )

    return graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    app = build_graph()
    try:
        diagram_bytes = app.get_graph().draw_mermaid_png()
        with open("graph_diagram.png", "wb") as f:
            f.write(diagram_bytes)
        print("[saved] graph_diagram.png")
    except Exception as exc:
        print(f"[warning] Could not generate png directly: {exc}")
        print("Mermaid diagram definition:")
        print(app.get_graph().draw_mermaid())