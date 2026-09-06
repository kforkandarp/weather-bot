from typing import Optional

import requests
from pydantic import BaseModel


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
REQUEST_TIMEOUT = 5


class LocationResult(BaseModel):
    name: str
    latitude: float
    longitude: float
    country: Optional[str] = None
    admin1: Optional[str] = None


def resolve_location(location: str) -> Optional[LocationResult]:
    """
    Resolve a place name using Open-Meteo geocoding.

    Open-Meteo may return multiple matches.
    We intentionally use the first result returned by the API.
    """

    if not location or not location.strip():
        return None

    params = {
        "name": location.strip(),
        "count": 10,
        "language": "en",
        "format": "json",
    }

    try:
        response = requests.get(
            GEOCODING_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        data = response.json()
        results = data.get("results", [])

        if not results:
            return None

        result = results[0]

        return LocationResult(
            name=result["name"],
            latitude=float(result["latitude"]),
            longitude=float(result["longitude"]),
            country=result.get("country"),
            admin1=result.get("admin1"),
        )

    except (
        requests.RequestException,
        ValueError,
        KeyError,
        TypeError,
    ):
        return None


def main() -> None:
    location = input("Enter a city: ").strip()

    result = resolve_location(location)

    if result is None:
        print("Location could not be resolved.")
        return

    print(result.model_dump())


if __name__ == "__main__":
    main()