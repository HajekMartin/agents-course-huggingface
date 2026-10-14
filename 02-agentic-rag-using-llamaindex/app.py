"""Run Alfred, a local LlamaIndex CodeAct agent with guest retrieval tools."""

import argparse
import asyncio
import os
import hashlib
import logging
import sys
from pathlib import Path
from urllib.parse import urlparse

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import requests
from llama_index.embeddings.ollama import OllamaEmbedding
from workflows.errors import WorkflowRuntimeError, WorkflowTimeoutError

from agent import create_llm, run_task
from diagnostics import configure_logging
from retriever import DATA_PATH, ROOT, GuestCatalog, GuestRetriever
from tools.guests import create_guest_tools
from tools.weather import get_weather
from tools.web import search_web

logger = logging.getLogger("alfred.app")


def validate_host(host: str) -> str:
    parsed = urlparse(host)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Use a local Ollama URL, e.g. http://localhost:11434.")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment or parsed.username:
        raise ValueError("Use the server root URL without a path or credentials.")
    return host.rstrip("/")


def check_models(host: str, *models: str):
    try:
        response = requests.get(f"{host}/api/tags", timeout=5)
        response.raise_for_status()
        installed = {model["name"]: model for model in response.json()["models"]}
    except (requests.RequestException, ValueError, KeyError, TypeError) as error:
        raise RuntimeError(f"Cannot reach Ollama at {host}. Start it with 'ollama serve'.") from error
    for model in models:
        name = model if ":" in model else f"{model}:latest"
        if name not in installed:
            raise RuntimeError(f"Model '{model}' is missing. Run 'ollama pull {model}'.")
        if model.endswith(":cloud"):
            raise ValueError("Select a locally downloaded model, not an Ollama cloud model.")
    return installed


def build_resources(args):
    host = validate_host(args.ollama_host)
    installed = check_models(host, args.model, args.embedding_model)
    catalog = GuestCatalog(args.data)
    prefix = args.embedding_model.split(":")[0] == "nomic-embed-text"
    embed = OllamaEmbedding(
        model_name=args.embedding_model, base_url=host,
        query_instruction="search_query:" if prefix else None,
        text_instruction="search_document:" if prefix else None,
        client_kwargs={"timeout": 180},
    )
    embedding_name = args.embedding_model if ":" in args.embedding_model else f"{args.embedding_model}:latest"
    retriever = GuestRetriever(catalog, embed, ROOT / "storage", installed[embedding_name].get("digest", ""))
    functions = create_guest_tools(catalog, retriever) + [get_weather]
    if args.enable_web:
        functions.append(search_web)
    logger.info("STARTUP pid=%s python=%s entrypoint=%s cwd=%s", os.getpid(), sys.executable, Path(__file__).resolve(), Path.cwd())
    source_files = sorted([*ROOT.glob("*.py"), *ROOT.glob("tools/*.py"), ROOT / "prompts.yaml"])
    fingerprint = hashlib.sha256(b"".join(path.read_bytes() for path in source_files)).hexdigest()[:12]
    logger.info("CONFIG code_version=%s model=%s embedding=%s ollama=%s", fingerprint, args.model, args.embedding_model, host)
    logger.info("DATA file=%s guests=%s index=%s", args.data.resolve(), len(catalog.guests), retriever.index_dir)
    logger.info("TOOLS web_enabled=%s tools=%s", args.enable_web, [function.__name__ for function in functions])
    if not args.enable_web:
        logger.warning("WEB DISABLED at startup. Enable the UI checkbox or restart with --enable-web.")
    return create_llm(args.model, host), functions, retriever


async def run_cli(llm, functions, task, max_iterations):
    async for answer, trace in run_task(llm, functions, task, max_iterations):
        if answer:
            print(f"\nAlfred: {answer}")


def launch_ui(llm, functions, port, max_iterations):
    import gradio as gr

    async def chat(message, history, enable_web):
        trace = ""
        logger.info("UI REQUEST checkbox_web_enabled=%s history_messages=%s", bool(enable_web), len(history or []))
        active_functions = [function for function in functions if function.__name__ != "search_web"]
        if enable_web:
            active_functions.append(search_web)
        try:
            async for answer, trace in run_task(llm, active_functions, message, max_iterations, history=history):
                yield answer or "Working…", trace
        except Exception as error:
            logger.exception("UI task failed")
            yield f"Task failed: {error}", trace

    with gr.Blocks(title="Alfred — Local Agentic RAG") as demo:
        web_enabled = gr.Checkbox(
            value=any(function.__name__ == "search_web" for function in functions),
            label="Enable web search (requires internet)",
        )
        web_enabled.change(
            fn=lambda enabled: logger.info("UI WEB TOGGLE web_enabled=%s", bool(enabled)),
            inputs=[web_enabled], outputs=None,
        )
        trace = gr.Textbox(label="Execution trace: model steps, code, and observations", lines=14)
        gr.ChatInterface(
            fn=chat, additional_inputs=[web_enabled], additional_outputs=[trace], title="Alfred — Local Agentic RAG",
            description="Explore the guestbook and draft invitations. Enable the checkbox above for web searches. Weather is simulated.",
            examples=[[example] for example in (
                "Who are my guests?",
                "Find guest what name is like Nicol",
                "Find guests interested in computers and programming.",
                "List all friends of the host, with their contact details.",
                "Draft an invitation for Ada Lovelace using her guest profile.",
                "Show simulated demo weather for Turnov.",
            )],
            cache_examples=False, save_history=True, concurrency_limit=1,
        )
    demo.launch(server_name="127.0.0.1", server_port=port, share=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "qwen2:7b"))
    parser.add_argument("--embedding-model", default=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"))
    parser.add_argument("--ollama-host", default=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--prompt", help="Run one task in the terminal instead of the chat UI.")
    parser.add_argument("--enable-web", action="store_true", help="Enable public web search (internet required).")
    parser.add_argument("--build-index", action="store_true", help="Build/load the local vector index and exit.")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    if not 2 <= args.max_iterations <= 30:
        parser.error("--max-iterations must be between 2 and 30.")
    try:
        llm, functions, retriever = build_resources(args)
        if args.build_index:
            retriever.get_index()
            print(f"Index ready: {retriever.index_dir}")
        elif args.prompt:
            asyncio.run(run_cli(llm, functions, args.prompt, args.max_iterations))
        else:
            launch_ui(llm, functions, args.port, args.max_iterations)
    except (OSError, ValueError, RuntimeError, WorkflowRuntimeError, WorkflowTimeoutError) as error:
        logger.error("Application failed: %s", error)
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
