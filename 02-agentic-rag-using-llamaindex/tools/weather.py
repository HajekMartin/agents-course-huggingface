"""Explicitly simulated weather, following the course's dummy tool."""

import random


def get_weather(location: str) -> dict:
    """Generate SIMULATED weather for a location; this is NOT a live forecast.

    Always tell the user that this is simulated course demo data. Do not present
    it as current weather or use it as evidence that a real event can go ahead.
    This tool does not fetch data from the internet and accepts no forecast date.

    Example task: Show demo weather for Turnov.
    <execute>
    result = get_weather(location="Turnov")
    print(result)
    </execute>
    Then describe the output explicitly as simulated weather.

    Args:
        location: The location label to include in the simulated output.
    """
    if not isinstance(location, str) or not location.strip():
        raise ValueError("Location must not be empty.")
    condition, temperature = random.choice([("Rainy", 15), ("Clear", 25), ("Windy", 20)])
    return {"location": location, "simulated": True, "condition": condition, "temp_c": temperature,
            "source": "Randomly generated course demo; not a live forecast"}
