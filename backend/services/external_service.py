import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY")

USE_LIVE_APIS = os.getenv("USE_LIVE_APIS", "false").lower() == "true"

TOMTOM_GEOCODE_URL = "https://api.tomtom.com/search/2/geocode"
TOMTOM_TRAFFIC_URL = (
    "https://api.tomtom.com/traffic/services/4/"
    "flowSegmentData/absolute/10/json"
)

OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"


def _geocode_place(place):
    """
    Convert a place name into latitude and longitude using TomTom.
    """
    if not TOMTOM_API_KEY:
        raise RuntimeError("TomTom API key is missing")

    url = f"{TOMTOM_GEOCODE_URL}/{place}.json"

    params = {
        "key": TOMTOM_API_KEY,
        "countrySet": "IN",
        "limit": 1
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()

    results = data.get("results", [])

    if not results:
        raise ValueError(f"Could not find coordinates for {place}")

    position = results[0]["position"]

    return float(position["lat"]), float(position["lon"])


def _weather_category(weather_main):
    """
    Convert OpenWeather conditions into the three
    categories used by our trained ML model.
    """

    weather_main = str(weather_main).lower()

    if weather_main == "clear":
        return "Sunny"

    if weather_main in {
        "rain",
        "drizzle",
        "thunderstorm",
        "snow"
    }:
        return "Rainy"

    return "Cloudy"


def _traffic_category(current_speed, free_flow_speed):
    """
    Convert TomTom traffic speed ratio into the
    three categories used by our ML model.

    >= 80% of free-flow speed  -> Low
    >= 50% of free-flow speed  -> Medium
    < 50%                      -> High
    """

    if free_flow_speed <= 0:
        return "Medium"

    ratio = current_speed / free_flow_speed

    if ratio >= 0.80:
        return "Low"

    if ratio >= 0.50:
        return "Medium"

    return "High"


def _historical_weather(place, month, slot, frame):
    rows = frame[frame["Location"] == place]

    if rows.empty:
        rows = frame

    row = rows.iloc[0]

    return {
        "weather_condition": row["Weather"],
        "temperature": float(row["Temperature"]),
        "source": "historical"
    }


def _historical_traffic(place, day, slot, frame):
    rows = frame[
        (frame["Location"] == place)
        & (frame["Day"] == day)
    ]

    if rows.empty:
        rows = frame[frame["Location"] == place]

    if rows.empty:
        rows = frame

    row = rows.iloc[0]

    return {
        "traffic_level": row["Traffic_Level"],
        "source": "historical"
    }


def _live_weather(place):
    """
    Get current weather from OpenWeather.
    """

    if not OPENWEATHER_API_KEY:
        raise RuntimeError("OpenWeather API key is missing")

    lat, lon = _geocode_place(place)

    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }

    response = requests.get(
        OPENWEATHER_URL,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    weather_main = data["weather"][0]["main"]
    temperature = float(data["main"]["temp"])

    return {
        "weather_condition": _weather_category(weather_main),
        "temperature": temperature,
        "source": "live",
        "latitude": lat,
        "longitude": lon
    }


def _live_traffic(place):
    """
    Get current traffic flow from TomTom.
    """

    if not TOMTOM_API_KEY:
        raise RuntimeError("TomTom API key is missing")

    lat, lon = _geocode_place(place)

    params = {
        "key": TOMTOM_API_KEY,
        "point": f"{lat},{lon}",
        "unit": "KMPH"
    }

    response = requests.get(
        TOMTOM_TRAFFIC_URL,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    flow = data["flowSegmentData"]

    current_speed = float(flow["currentSpeed"])
    free_flow_speed = float(flow["freeFlowSpeed"])

    traffic_level = _traffic_category(
        current_speed,
        free_flow_speed
    )

    return {
        "traffic_level": traffic_level,
        "current_speed": current_speed,
        "free_flow_speed": free_flow_speed,
        "source": "live",
        "latitude": lat,
        "longitude": lon
    }


def weather_for(place, month, slot, frame):
    """
    Main weather function.

    Uses OpenWeather when live APIs are enabled.
    Falls back to historical data if live API fails.
    """

    if USE_LIVE_APIS:
        try:
            return _live_weather(place)
        except Exception as error:
            print(f"Live weather failed for {place}: {error}")

            result = _historical_weather(
                place,
                month,
                slot,
                frame
            )

            result["source"] = "historical_fallback"
            return result

    return _historical_weather(
        place,
        month,
        slot,
        frame
    )


def traffic_for(place, day, slot, frame):
    """
    Main traffic function.

    Uses TomTom when live APIs are enabled.
    Falls back to historical data if live API fails.
    """

    if USE_LIVE_APIS:
        try:
            return _live_traffic(place)
        except Exception as error:
            print(f"Live traffic failed for {place}: {error}")

            result = _historical_traffic(
                place,
                day,
                slot,
                frame
            )

            result["source"] = "historical_fallback"
            return result

    return _historical_traffic(
        place,
        day,
        slot,
        frame
    )