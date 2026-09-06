from langchain_groq import ChatGroq

from app.config import GROQ_API_KEY, GROQ_MODEL


def get_llm() -> ChatGroq:
    """
    Create and return the Groq chat model used by the application.
    """

    return ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,
    )