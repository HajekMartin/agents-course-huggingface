"""Run the Hugging Face first-agent tutorial locally with Ollama."""

import argparse
import os
import random
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# Avoid background analytics and remote model metadata lookups.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import requests
import yaml
from smolagents import CodeAgent, FinalAnswerTool, LiteLLMModel, tool
from tzlocal import get_localzone


@tool
def roll_dice(min_rolls: int = 1, max_rolls: int = 6) -> dict:
    """Choose a random number of rolls, then roll a six-sided die that many times.

    Use roll_dice whenever the user asks for dice rolls or random dice results.
    The tool randomly chooses the number of rolls as well as every die result.
    It returns a dictionary with count, rolls, and total keys.
    For a random number of rolls without specified bounds, call roll_dice() once.
    When bounds are specified, pass them as min_rolls and max_rolls.
    For an exact number of rolls, set min_rolls and max_rolls to that same number.
    Report the actual count, all individual rolls, and the total from the tool.
    Never invent random results or reroll to obtain a preferred result.
    Do not implement dice rolling yourself or import random: use roll_dice.

    Example task: Give me 5 rounds of dice.
    Thought: I will roll exactly five times and return the tool's result.
    <code>
    result = roll_dice(min_rolls=5, max_rolls=5)
    final_answer(result)
    </code>

    Example task: Roll a random number of dice between 2 and 10 times.
    Thought: I will call the dice tool once to get the actual results.
    <code>
    result = roll_dice(min_rolls=2, max_rolls=10)
    print(result)
    </code>
    After the runtime returns the observation, finish with:
    Thought: I will return the count, rolls, and total from the tool.
    <code>
    final_answer(result)
    </code>

    Args:
        min_rolls: Minimum number of rolls, inclusive; at least 1.
        max_rolls: Maximum number of rolls, inclusive; at most 1000.

    Returns:
        A dictionary containing the randomly chosen count, individual rolls, and total.
    """
    if type(min_rolls) is not int or type(max_rolls) is not int:
        raise ValueError("Roll bounds must be integers.")
    if not 1 <= min_rolls <= max_rolls <= 1000:
        raise ValueError("Roll bounds must satisfy 1 <= min_rolls <= max_rolls <= 1000.")

    count = random.randint(min_rolls, max_rolls)
    rolls = [random.randint(1, 6) for _ in range(count)]
    return {"count": count, "rolls": rolls, "total": sum(rolls)}


@tool
def get_local_timezone() -> dict:
    """Get the timezone configured on the computer running this local application.

    For "my current timezone", "local timezone", or the current time without a
    specified timezone, call get_local_timezone(). It reads this computer's settings.
    Never guess the local timezone from the user's language or from a UTC offset.
    Report the timezone name and current UTC offset returned by the tool.

    Example task: What is my current timezone?
    Thought: I will read the timezone configured on this computer.
    <code>
    result = get_local_timezone()
    final_answer(result)
    </code>

    Returns:
        The system timezone name, current UTC offset, abbreviation, and local time.
        This describes this computer, not the location of a remote browser user.
    """
    try:
        timezone = get_localzone()
        local_time = datetime.now(timezone)
    except (OSError, ValueError, ZoneInfoNotFoundError) as error:
        return {"error": f"Cannot detect this computer's timezone: {error}"}
    offset = local_time.strftime("%z")
    return {
        "timezone": timezone.key,
        "utc_offset": f"{offset[:3]}:{offset[3:]}",
        "abbreviation": local_time.tzname(),
        "local_time": local_time.isoformat(timespec="seconds"),
        "source": "Timezone configured on this computer",
    }


@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """Get the current local time in an IANA timezone without a network request.

    Use get_current_time_in_timezone for current time questions with a specified
    timezone. For local time without a specified timezone, use get_local_timezone.

    Example task: What time is it in Europe/Prague?
    Thought: I will get the current time from the local timezone tool.
    <code>
    result = get_current_time_in_timezone(timezone="Europe/Prague")
    print(result)
    </code>
    After the observation, use final_answer(result) in a code block.

    Args:
        timezone: IANA timezone name, such as 'Europe/Prague' or 'America/New_York'.
    """
    try:
        local_time = datetime.now(ZoneInfo(timezone))
    except (ZoneInfoNotFoundError, ValueError):
        return f"Unknown timezone: {timezone}. Use an IANA name such as Europe/Prague."
    return f"The current local time in {timezone} is: {local_time:%Y-%m-%d %H:%M:%S %Z (%z)}"


def validate_ollama_host(host: str) -> str:
    """Keep inference on the local Ollama server."""
    parsed = urlparse(host)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
        "localhost", "127.0.0.1", "::1"
    }:
        raise ValueError("Ollama host must be a local URL, e.g. http://localhost:11434.")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment or parsed.username:
        raise ValueError("Use the Ollama server root URL without a path or credentials.")
    return host.rstrip("/")


def check_ollama(host: str, model_id: str) -> None:
    """Report a helpful startup error if Ollama or the selected model is missing."""
    try:
        response = requests.get(f"{host}/api/tags", timeout=5)
        response.raise_for_status()
        models = {entry["name"] for entry in response.json()["models"]}
    except (requests.RequestException, ValueError, KeyError, TypeError) as error:
        raise RuntimeError(
            f"Cannot reach Ollama at {host}. Start it with 'ollama serve'."
        ) from error
    name = model_id if ":" in model_id else f"{model_id}:latest"
    if name not in models:
        installed = ", ".join(sorted(models)) or "none"
        raise RuntimeError(
            f"Model '{model_id}' is not installed. Run 'ollama pull {model_id}' "
            f"or select an installed model with --model. Installed models: {installed}."
        )


def create_agent(model_id: str, host: str) -> CodeAgent:
    """Build the tutorial's CodeAgent using only local tools and local inference."""
    host = validate_ollama_host(host)
    with Path(__file__).with_name("prompts.yaml").open(encoding="utf-8") as stream:
        local_prompts = yaml.safe_load(stream)
    # Keep framework templates for planning/final answers while customizing the system prompt.
    bundled_prompts = files("smolagents").joinpath("prompts/code_agent.yaml")
    prompt_templates = yaml.safe_load(bundled_prompts.read_text(encoding="utf-8"))
    prompt_templates["system_prompt"] = local_prompts["system_prompt"]

    model = LiteLLMModel(
        model_id=f"ollama_chat/{model_id}",
        api_base=host,
        temperature=0.2,
        max_tokens=2096,
        num_ctx=16384,
        timeout=180,
    )
    return CodeAgent(
        model=model,
        tools=[FinalAnswerTool(), roll_dice, get_local_timezone, get_current_time_in_timezone],
        max_steps=6,
        verbosity_level=1,
        planning_interval=None,
        prompt_templates=prompt_templates,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "qwen2:7b"))
    parser.add_argument(
        "--ollama-host", default=os.getenv("OLLAMA_HOST", "http://localhost:11434")
    )
    parser.add_argument("--prompt", help="Run one task in the terminal instead of the web UI.")
    parser.add_argument("--port", type=int, default=7860, help="Local Gradio port (default: 7860).")
    args = parser.parse_args()
    try:
        host = validate_ollama_host(args.ollama_host)
        check_ollama(host, args.model)
        agent = create_agent(args.model, host)
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")

    if args.prompt:
        print(agent.run(args.prompt))
    else:
        from Gradio_UI import GradioUI

        GradioUI(agent).launch(server_name="127.0.0.1", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
