"""LlamaIndex CodeActAgent with local Ollama and a fresh executor per task."""

from pathlib import Path
import re
import logging
from time import perf_counter
from uuid import uuid4

import yaml
from llama_index.core.agent.workflow import AgentInput, CodeActAgent, ToolCall, ToolCallResult
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage

from executor import CodeExecutor
from diagnostics import preview, request_id

logger = logging.getLogger("alfred.agent")

ROOT = Path(__file__).resolve().parent

GUEST_TOOLS = {"list_guests", "find_guests_by_name", "search_guests", "get_guest", "filter_guests"}


def guard_guest_answer(answer: str, calls: list) -> str:
    """Do not let a model turn an empty result or template into invented profiles."""
    successful = [call for call in calls if isinstance(call.get("result"), dict) and "error" not in call["result"]]
    if re.search(r"\bborn\b|\bbirth\b|narozen", answer, flags=re.IGNORECASE):
        years = set(re.findall(r"\b(?:1\d{3}|20\d{2})\b", answer))
        evidence = " ".join(str(call["result"]) for call in successful)
        if years - set(re.findall(r"\b(?:1\d{3}|20\d{2})\b", evidence)):
            return "The retrieved sources do not support those birth years. The local guestbook does not provide them; a web lookup with supporting sources is needed."
    guest_calls = [call for call in successful if call["tool"] in GUEST_TOOLS]
    if not guest_calls:
        return answer
    records = {}
    for call in guest_calls:
        result = call["result"]
        rows = result.get("guests", result.get("candidates", []))
        if result.get("guest"):
            rows = [result["guest"]]
        for row in rows:
            records[row["id"]] = row
    if not records and len(guest_calls) == len(successful):
        if all(call["tool"] == "search_guests" for call in guest_calls):
            return "No guest candidates exceeded the similarity cutoff for these searches. This does not prove that no relevant guest exists."
        return "No matching guests were found by the guestbook lookups performed."
    if re.search(r"\[(?:Candidate Profile\s*\d+|Score)\]", answer, flags=re.IGNORECASE):
        return "Retrieved guest records:\n\n" + "\n\n".join(
            f"- **{row['name']}**: {row['description']}\n  Source: `{row['source']}`"
            for row in records.values()
        )
    missing_sources = [row["source"] for row in records.values() if row["source"] not in answer]
    if missing_sources:
        answer += "\n\nGuest sources consulted: " + ", ".join(f"`{source}`" for source in missing_sources)
    return answer


def create_agent(llm, functions):
    executor = CodeExecutor(functions)
    prompt = yaml.safe_load((ROOT / "prompts.yaml").read_text(encoding="utf-8"))["system_prompt"]
    agent = CodeActAgent(
        name="Alfred",
        llm=llm,
        tools=list(functions),
        code_execute_fn=executor.execute,
        code_act_system_prompt=prompt,
        streaming=False,
    )
    return agent, executor


def create_llm(model: str, host: str):
    return Ollama(
        model=model, base_url=host, temperature=0.0, request_timeout=180,
        context_window=16384, additional_kwargs={"num_predict": 2048},
        is_function_calling_model=False,
    )


def conversation_messages(history):
    """Bound textual chat context; never reuse code execution state or trace logs."""
    messages = []
    for entry in (history or [])[-8:]:
        if entry.get("role") not in {"user", "assistant"}:
            continue
        content = entry.get("content")
        if isinstance(content, list):
            content = "\n".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
        if isinstance(content, str) and content.strip() and content != "Working…":
            messages.append(ChatMessage(role=entry["role"], content=content[-2000:]))
    return messages


async def run_task(llm, functions, task: str, max_iterations: int = 8, history=None):
    """Log one task's active capabilities and lifecycle, including cancellations."""
    token = request_id.set(uuid4().hex[:8])
    started = perf_counter()
    names = [function.__name__ for function in functions]
    logger.info("TASK START model=%s web_enabled=%s tools=%s history_messages=%s max_iterations=%s task=%s",
                getattr(llm, "model", type(llm).__name__), "search_web" in names, names,
                len(conversation_messages(history)), max_iterations, preview(task, 500))
    try:
        async for answer, trace in _run_task(llm, functions, task, max_iterations, history):
            if answer:
                logger.info("FINAL ANSWER %s", preview(answer))
            yield answer, trace
    except Exception:
        logger.exception("TASK FAILED")
        raise
    finally:
        logger.info("TASK END duration=%.2fs", perf_counter() - started)
        request_id.reset(token)


async def _run_task(llm, functions, task: str, max_iterations: int = 8, history=None):
    """Yield answer/trace pairs; trace records model steps and actual code execution."""
    if not task.strip():
        raise ValueError("Task must not be empty.")
    agent, executor = create_agent(llm, functions)
    trace = ""
    step = 0
    messages = conversation_messages(history)
    has_web = any(function.__name__ == "search_web" for function in functions)
    explicit_web_request = bool(re.search(
        r"\b(search|find|verify|check|look up|vyhledej|najdi|dohledej|ověř)\b.*\b(internet|online|web|internetu|webu)\b"
        r"|\b(web search|internet search)\b", task, flags=re.IGNORECASE
    )) and not re.search(r"\b(do not|don't|without|never)\s+(search|use|access)", task, flags=re.IGNORECASE)
    current_task = task
    logger.debug("CONVERSATION CONTEXT %s", preview(messages, 8000))
    logger.debug("REGISTERED TOOL DESCRIPTIONS %s", agent._get_tool_descriptions(agent.tools))
    if explicit_web_request and not has_web:
        logger.warning("WEB REQUESTED BUT DISABLED: search_web is absent from this task's prompt and executor.")
    for attempt in range(2):
        handler = agent.run(user_msg=current_task, chat_history=messages, max_iterations=max_iterations - step)
        async for event in handler.stream_events():
            if isinstance(event, AgentInput):
                step += 1
                logger.info("MODEL STEP %s attempt=%s", step, attempt + 1)
                trace += f"Step {step}: model request\n"
            elif isinstance(event, ToolCall):
                logger.info("EXECUTION REQUEST tool=%s code=%s", event.tool_name, preview(event.tool_kwargs.get("code", event.tool_kwargs)))
                trace += f"\nExecuting {event.tool_name}:\n{event.tool_kwargs.get('code', str(event.tool_kwargs))}\n"
            elif isinstance(event, ToolCallResult):
                trace += f"\nObservation:\n{event.tool_output}\n\n"
            else:
                continue
            yield "", trace
        output = await handler
        raw_answer = output.response.content or "The agent returned no text answer."
        logger.info("MODEL ANSWER BEFORE CHECKS %s", preview(raw_answer))
        answer = guard_guest_answer(raw_answer, executor.calls)
        if answer != raw_answer:
            logger.info("ANSWER CHECK changed the model answer")
        searched = any(call["tool"] == "search_web" for call in executor.calls)
        logger.info("WEB STATUS enabled=%s attempted=%s calls=%s", has_web, searched,
                    sum(call["tool"] == "search_web" for call in executor.calls))
        missing_web = has_web and not searched and (explicit_web_request or answer != raw_answer and "birth years" in answer)
        if missing_web and attempt == 0 and step + 2 <= max_iterations:
            logger.warning("WEB TOOL SKIPPED by model; starting one corrective attempt")
            trace += "Verification: no web search was executed. Requesting one corrective attempt.\n"
            messages += [ChatMessage(role="user", content=task), ChatMessage(role="assistant", content=raw_answer)]
            current_task = (
                "The previous answer was not verified online. search_web is available. "
                "You MUST execute search_web using Python in <execute> tags, print the results, "
                "and answer only from the returned snippets with their source URLs. "
                "Use the preceding conversation to resolve names and references. Original request: " + task
            )
            yield "", trace
            continue
        if missing_web:
            logger.error("WEB TOOL NOT EXECUTED after correction or iteration budget exhausted")
            answer = "No web search was executed, so an online-verified answer is not available."
        break
    if executor.calls and any("error" in call for call in executor.calls):
        trace += "Some tool calls failed; see observations above.\n"
    yield answer, trace
