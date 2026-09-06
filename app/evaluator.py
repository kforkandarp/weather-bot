from typing import Any

from app.state import SOP, SOPCondition, UserIntent, WeatherState


SUPPORTED_OPERATORS = {
    "==",
    "!=",
    ">",
    ">=",
    "<",
    "<=",
    "between",
    "in",
    "contains",
}


SEVERITY_RANK = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


def build_context(
    intent: UserIntent,
    weather: WeatherState,
) -> dict[str, Any]:
    """
    Combine user intent and live weather facts into one evaluation context.

    Weather values always come from WeatherState, which itself is populated
    from Open-Meteo. The user's message cannot override those weather facts.
    """

    return {
        # User-derived fields
        "activity": intent.activity,
        "location": intent.location,
        "time_expression": intent.time_expression,
        "vulnerable_group": intent.vulnerable_group,
        "hour": intent.hour if intent.hour is not None else weather.hour,

        # API-derived weather facts
        "temperature": weather.temperature,
        "wind_speed": weather.wind_speed,
        "precipitation": weather.precipitation,
        "precipitation_probability": weather.precipitation_probability,
        "uv_index": weather.uv_index,
    }


def _compare(
    actual: Any,
    operator: str,
    expected: Any,
) -> bool:
    """
    Apply one generic policy comparison.

    Returns:
        bool: Whether the condition passes.
    """

    if operator not in SUPPORTED_OPERATORS:
        raise ValueError(
            f"Unsupported SOP operator: {operator}"
        )

    if actual is None:
        return False

    try:
        if operator == "==":
            return actual == expected

        if operator == "!=":
            return actual != expected

        if operator == ">":
            return actual > expected

        if operator == ">=":
            return actual >= expected

        if operator == "<":
            return actual < expected

        if operator == "<=":
            return actual <= expected

        if operator == "between":
            if not isinstance(expected, list) or len(expected) != 2:
                raise ValueError(
                    "'between' requires [lower, upper]"
                )

            lower, upper = expected

            return lower <= actual <= upper

        if operator == "in":
            if not isinstance(expected, list):
                raise ValueError(
                    "'in' requires a list of values"
                )

            return actual in expected

        if operator == "contains":
            return expected in actual

    except (TypeError, ValueError):
        return False

    return False


def evaluate_condition(
    condition: SOPCondition,
    context: dict[str, Any],
) -> bool:
    """
    Evaluate one SOP condition against the current context.
    """

    actual = context.get(condition.field)

    return _compare(
        actual=actual,
        operator=condition.operator,
        expected=condition.value,
    )


def evaluate_boolean_sop(
    sop: SOP,
    context: dict[str, Any],
) -> bool:
    """
    Evaluate an SOP using its boolean condition structure.

    Semantics:

        all AND (any)

    More precisely:
        - Every condition in `all` must pass.
        - If `any` contains conditions, at least one must pass.
        - If `any` is empty, the any-group automatically passes.
    """

    if sop.conditions is None:
        return False

    all_conditions = sop.conditions.all
    any_conditions = sop.conditions.any

    all_pass = all(
        evaluate_condition(
            condition,
            context,
        )
        for condition in all_conditions
    )

    if not all_pass:
        return False

    if not any_conditions:
        return True

    return any(
        evaluate_condition(
            condition,
            context,
        )
        for condition in any_conditions
    )


def evaluate_fuzzy_sop(
    sop: SOP,
    context: dict[str, Any],
) -> bool:
    """
    Evaluate a fuzzy SOP using weighted criteria.

    Required conditions must all pass first.

    Each satisfied criterion contributes its configured weight.
    The SOP applies when the final score reaches score_threshold.
    """

    required_pass = all(
        evaluate_condition(
            condition,
            context,
        )
        for condition in sop.required_conditions
    )

    if not required_pass:
        return False

    if not sop.criteria:
        return False

    score = 0.0

    for criterion in sop.criteria:
        if evaluate_condition(
            criterion,
            context,
        ):
            score += criterion.weight or 1.0

    threshold = sop.score_threshold

    if threshold is None:
        return False

    return score >= threshold


def evaluate_sop(
    sop: SOP,
    intent: UserIntent,
    weather: WeatherState,
) -> bool:
    """
    Evaluate one SOP against user intent and live weather.

    This function is completely policy-driven. It does not contain
    SOP-specific IDs or activity-specific branches.
    """

    context = build_context(
        intent=intent,
        weather=weather,
    )

    if sop.condition_type == "boolean":
        return evaluate_boolean_sop(
            sop=sop,
            context=context,
        )

    if sop.condition_type == "fuzzy":
        return evaluate_fuzzy_sop(
            sop=sop,
            context=context,
        )

    return False


def evaluate_sops(
    sops: list[SOP],
    intent: UserIntent,
    weather: WeatherState,
) -> list[SOP]:
    """
    Return every SOP whose conditions actually pass.

    Semantic retrieval determines candidates.

    This function determines actual applicability.
    """

    applicable = []

    for sop in sops:
        if evaluate_sop(
            sop=sop,
            intent=intent,
            weather=weather,
        ):
            applicable.append(sop)

    return applicable


def rank_sops(sops: list[SOP]) -> list[SOP]:
    """
    Deterministically rank applicable SOPs.

    Resolution order:
        1. Severity descending
        2. Priority descending
        3. SOP ID ascending for exact ties
    """

    return sorted(
        sops,
        key=lambda sop: (
            -SEVERITY_RANK[sop.severity],
            -sop.priority,
            sop.id,
        ),
    )


def select_sop(sops: list[SOP]) -> SOP | None:
    """
    Select the winning SOP from applicable policies.

    Returns None when no SOP applies.
    """

    ranked = rank_sops(sops)

    if not ranked:
        return None

    return ranked[0]


def main() -> None:
    """Run deterministic evaluator checks using manually constructed data."""

    intent = UserIntent(
        activity="cycling",
        location="Bhopal",
        time_expression="today",
    )

    weather = WeatherState(
        location="Bhopal",
        latitude=23.25469,
        longitude=77.40289,
        timestamp="2026-09-06T17:00",
        hour=17,
        temperature=30.0,
        wind_speed=45.0,
        precipitation=0.1,
        precipitation_probability=20.0,
        uv_index=2.0,
    )

    from app.sop_loader import load_sops

    sops = load_sops()

    applicable = evaluate_sops(
        sops=sops,
        intent=intent,
        weather=weather,
    )

    selected = select_sop(applicable)

    print("Applicable SOPs:")

    for sop in applicable:
        print(
            f"- {sop.id} | "
            f"{sop.severity} | "
            f"priority={sop.priority}"
        )

    print("\nSelected SOP:")

    if selected is None:
        print("None")
    else:
        print(
            f"{selected.id} | "
            f"{selected.severity} | "
            f"{selected.advice}"
        )


if __name__ == "__main__":
    main()