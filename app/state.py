from typing import Any, Literal, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from pydantic import BaseModel, Field

class UserIntent(BaseModel):
    """Structured interpretation of the user's request."""

    activity: Optional[str] = Field(
        default=None,
        description=(
            "The outdoor activity the user is asking about. "
            "Examples: cycling, biking, running, walking, picnic, "
            "park, travel, commute, driving."
        ),
    )

    location: Optional[str] = Field(
        default=None,
        description="City or place where the activity will happen.",
    )

    time_expression: Optional[str] = Field(
        default=None,
        description=(
            "The user's requested time period, such as today, "
            "now, this evening, tomorrow, or a specific time."
        ),
    )

    hour: Optional[int] = Field(
        default=None,
        description=(
            "Specific local hour if the user explicitly gives one. "
            "Use 24-hour format. Example: 4 PM -> 16."
        ),
    )

    vulnerable_group: Optional[str] = Field(
        default=None,
        description=(
            "Relevant vulnerable group if explicitly mentioned. "
            "Examples: child, children, elderly, senior, pet, dog."
        ),
    )


# -----------------------------
# WEATHER DATA
# -----------------------------

class WeatherState(BaseModel):
    """Weather facts obtained from Open-Meteo."""

    location: str
    latitude: float
    longitude: float

    timestamp: str
    hour: int

    temperature: float
    wind_speed: float
    precipitation: float
    precipitation_probability: float
    uv_index: float

    temperature_unit: str = "°C"
    wind_speed_unit: str = "km/h"
    precipitation_unit: str = "mm"
    source: str = "Open-Meteo"


# -----------------------------
# SOP MODELS
# -----------------------------

class SOPCondition(BaseModel):
    """One deterministic condition inside an SOP."""

    field: str
    operator: str
    value: Any
    weight: Optional[float] = None


class BooleanConditions(BaseModel):
    """
    Boolean policy logic.

    all = every condition must pass.
    any = at least one condition must pass.

    Therefore:
        all AND (any condition)
    """

    all: list[SOPCondition] = Field(default_factory=list)
    any: list[SOPCondition] = Field(default_factory=list)


class SOP(BaseModel):
    """One independently maintained policy."""

    id: str
    category: str
    severity: Literal["low", "medium", "high", "critical"]
    priority: int
    description: str

    condition_type: Literal["boolean", "fuzzy"]

    conditions: Optional[BooleanConditions] = None

    required_conditions: list[SOPCondition] = Field(
        default_factory=list
    )

    criteria: list[SOPCondition] = Field(
        default_factory=list
    )

    score_threshold: Optional[float] = None

    advice: str


# -----------------------------
# GRAPH STATE
# -----------------------------

class AgentState(TypedDict, total=False):
    

    # Conversation/session
    messages: Annotated[list[BaseMessage], add_messages]

    # Original user input
    user_query: str

    # Structured interpretation
    intent: Optional[UserIntent]

    # Resolved location
    resolved_location: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]

    # Live weather
    weather: Optional[WeatherState]

    # SOP retrieval
    candidate_sops: list[SOP]

    # Deterministic policy evaluation
    applicable_sops: list[SOP]
    selected_sop: Optional[SOP]

    # Outcome / routing information
    outcome: Optional[
        Literal[
            "SUCCESS",
            "MISSING_LOCATION",
            "LOCATION_RESOLUTION_FAILED",
            "WEATHER_FETCH_FAILED",
            "NO_APPLICABLE_SOP",
        ]
    ]

    # Final user-facing answer
    final_response: Optional[str]