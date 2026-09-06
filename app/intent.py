import re
from typing import Optional

from app.llm import get_llm
from app.state import UserIntent


def _normalize_activity(activity: Optional[str]) -> Optional[str]:
    if not activity:
        return None

    value = activity.strip().lower()

    aliases = {
        "bike": "cycling",
        "bicycle": "cycling",
        "biking": "cycling",
        "ride": "cycling",
        "bike ride": "cycling",
        "cycling": "cycling",
        "run": "running",
        "jog": "running",
        "jogging": "running",
        "walk": "walking",
        "going for a walk": "walking",
        "picnic": "picnic",
        "commuting": "commute",
        "commute": "commute",
        "driving": "driving",
        "drive": "driving",
        "travel": "travel",
        "travelling": "travel",
        "traveling": "travel",
        "park": "park",
        "outdoor play": "outdoor_play",
        "outdoor activity": "outdoor_activity",
        "outdoor exercise": "outdoor_exercise",
    }

    return aliases.get(value, value)


def _normalize_vulnerable_group(
    vulnerable_group: Optional[str],
) -> Optional[str]:
    if not vulnerable_group:
        return None

    value = vulnerable_group.strip().lower()

    aliases = {
        "kid": "child",
        "kids": "children",
        "child": "child",
        "children": "children",
        "childrens": "children",
        "elder": "elderly",
        "elderly": "elderly",
        "senior": "senior",
        "senior citizen": "senior",
        "older adult": "older_adult",
        "pet": "pet",
        "pets": "pet",
        "dog": "dog",
        "dogs": "dog",
    }

    return aliases.get(value, value)


def _parse_explicit_hour(text: str) -> Optional[int]:
    pattern = (
        r"\b(1[0-2]|0?[1-9])"
        r"(?:\s*:\s*[0-5]\d)?"
        r"\s*(a\.?m\.?|p\.?m\.?)\b"
    )

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    hour = int(match.group(1))
    meridiem = match.group(2).lower().replace(".", "")

    if meridiem == "am":
        return 0 if hour == 12 else hour

    return 12 if hour == 12 else hour + 12


def _normalize_time_expression(
    query: str,
    extracted: Optional[str],
) -> tuple[Optional[str], Optional[int]]:
    hour = _parse_explicit_hour(query)

    if hour is not None:
        return extracted or query, hour

    return extracted, None


def _build_extraction_prompt(
    user_query: str,
    previous_intent: Optional[UserIntent],
) -> str:
    previous_context = "None"

    if previous_intent is not None:
        previous_context = previous_intent.model_dump_json()

    return f"""
You extract structured intent from a weather-advisory chat.

Current user message:
{user_query}

Previous structured intent:
{previous_context}

Extract:
- activity
- location
- time_expression
- hour
- vulnerable_group

Rules:

1. Extract information explicitly present in the current message.

2. For a follow-up message, preserve information from the previous
   intent ONLY when the current message clearly refers to that
   previous context.

3. Location can be inherited when the user is clearly continuing
   the previous weather question.

4. Activity can be inherited when the user is clearly continuing
   the previous activity.

5. Time can be inherited when the user is clearly continuing the
   previous question.

6. Vulnerable group is different:
   NEVER carry a previous vulnerable group into a new standalone
   activity or question.

   Only preserve a previous vulnerable group when the current
   message clearly refers back to that person/group.

   Example:
   Previous: "Should I take my kid to the park?"
   Current: "What about tomorrow?"
   -> preserve child.

   Previous: "Should I take my kid to the park?"
   Current: "Would today be good for a picnic?"
   -> vulnerable_group must be null.

7. Do not invent a location.

8. Do not invent an activity.

9. Do not infer weather conditions.

10. If something is genuinely unknown, return null.

11. If an explicit clock time is present, extract it.

12. Do not answer the user. Return structured information only.
""".strip()


def extract_intent(
    user_query: str,
    previous_intent: Optional[UserIntent] = None,
) -> UserIntent:

    llm = get_llm()

    structured_llm = llm.with_structured_output(
        UserIntent,
        method="json_schema",
    )

    prompt = _build_extraction_prompt(
        user_query=user_query,
        previous_intent=previous_intent,
    )

    intent = structured_llm.invoke(prompt)

    intent.activity = _normalize_activity(
        intent.activity
    )

    intent.vulnerable_group = _normalize_vulnerable_group(
        intent.vulnerable_group
    )

    (
        intent.time_expression,
        deterministic_hour,
    ) = _normalize_time_expression(
        query=user_query,
        extracted=intent.time_expression,
    )

    if deterministic_hour is not None:
        intent.hour = deterministic_hour

    return intent


def main() -> None:
    queries = [
        "Is it safe to cycle in Bhopal today?",
        "Should I take my kid to the park today?",
        "Would today be good for a picnic?",
        "Is it safe to go cycling at 4 PM in Bhopal?",
        "Can I take my dog for a walk tomorrow at 7 AM?",
    ]

    previous_intent = None

    for query in queries:
        print(f"\nQuery: {query}")

        intent = extract_intent(
            user_query=query,
            previous_intent=previous_intent,
        )

        print(intent.model_dump())

        previous_intent = intent


if __name__ == "__main__":
    main()