from datetime import datetime, timedelta
from typing import Optional
import requests
from app.location import LocationResult
from app.state import WeatherState

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT = 10
MAX_RETRIES = 3


def _parse_api_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _find_hourly_weather(hourly: dict, target_time: datetime) -> Optional[dict]:
    times = hourly.get("time", [])
    if not times:
        return None

    required_fields = [
        "temperature_2m",
        "wind_speed_10m",
        "precipitation",
        "precipitation_probability",
        "uv_index",
    ]

    for field in required_fields:
        if field not in hourly:
            return None

    best_index = min(
        range(len(times)),
        key=lambda idx: abs(_parse_api_time(times[idx]) - target_time),
    )

    return {
        "time": times[best_index],
        "temperature_2m": hourly["temperature_2m"][best_index],
        "wind_speed_10m": hourly["wind_speed_10m"][best_index],
        "precipitation": hourly["precipitation"][best_index],
        "precipitation_probability": hourly["precipitation_probability"][best_index],
        "uv_index": hourly["uv_index"][best_index],
    }


def _choose_target_time(
    current_time: datetime,
    time_expression: Optional[str],
    requested_hour: Optional[int],
) -> datetime:
    expression = (time_expression or "").lower()
    target_date = current_time.date()

    if "tomorrow" in expression:
        target_date += timedelta(days=1)

    hour = current_time.hour
    if requested_hour is not None:
        hour = requested_hour
    elif "evening" in expression or "night" in expression:
        hour = 18
    elif "afternoon" in expression:
        hour = 14
    elif "morning" in expression:
        hour = 9

    return datetime.combine(target_date, datetime.min.time()).replace(hour=hour)


def fetch_weather(
    location: LocationResult,
    time_expression: Optional[str] = None,
    requested_hour: Optional[int] = None,
) -> Optional[WeatherState]:
    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "current": (
            "temperature_2m,"
            "wind_speed_10m,"
            "precipitation,"
            "precipitation_probability,"
            "uv_index"
        ),
        "hourly": (
            "temperature_2m,"
            "wind_speed_10m,"
            "precipitation,"
            "precipitation_probability,"
            "uv_index"
        ),
        "forecast_days": 2,
        "timezone": "auto",
    }

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(FORECAST_URL, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()

            current = data.get("current")
            hourly = data.get("hourly")
            if not current or not hourly:
                raise ValueError("Open-Meteo response missing current/hourly weather data.")

            current_time = _parse_api_time(current["time"])
            target_time = _choose_target_time(
                current_time=current_time,
                time_expression=time_expression,
                requested_hour=requested_hour,
            )

            weather = _find_hourly_weather(hourly=hourly, target_time=target_time)
            if weather is None:
                raise ValueError("Required hourly weather fields were missing.")

            return WeatherState(
                location=location.name,
                latitude=location.latitude,
                longitude=location.longitude,
                timestamp=weather["time"],
                hour=_parse_api_time(weather["time"]).hour,
                temperature=float(weather["temperature_2m"]),
                wind_speed=float(weather["wind_speed_10m"]),
                precipitation=float(weather["precipitation"]),
                precipitation_probability=float(weather["precipitation_probability"]),
                uv_index=float(weather["uv_index"]),
            )
        except Exception as exc:
            last_error = exc

    print(f"[weather] Open-Meteo request failed after {MAX_RETRIES} attempts: {last_error}")
    return None