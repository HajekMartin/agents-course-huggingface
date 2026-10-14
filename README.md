# Agents Course — Local Ollama Projects

Hands-on projects based on the
[Hugging Face AI Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction),
adapted to run with locally downloaded models through Ollama.

The repository explores tools, code execution, agent steps, and retrieval-augmented
generation (RAG). Each project is independent, with its own dependencies, virtual
environment, configuration, tests, and English documentation.

## Projects

| Project | Framework | Capabilities | Local UI |
| --- | --- | --- | --- |
| [01 — First Agent](01-first-agent-using-smolagents/README.md) | smolagents `CodeAgent` | Random dice rolls, local timezone detection, and time in a specified timezone | http://127.0.0.1:7860 |
| [02 — Alfred: Agentic RAG](02-agentic-rag-using-llamaindex/README.md) | LlamaIndex `CodeActAgent` | Guest listing, exact lookup and filtering, approximate name matching, semantic search, optional web search, and simulated weather | http://127.0.0.1:7861 |

Both agents generate Python calls to prepared tools and use execution results
to decide whether to continue or return an answer. Tool-specific instructions
and examples live beside the tool definitions and are included automatically
in the model prompt.

## Requirements

- Python 3.10 or newer.
- Ollama installed and running locally.
- A local language model; both projects default to `qwen2:7b`.
- For the second project, the local embedding model `nomic-embed-text`.

```bash
ollama pull qwen2:7b
ollama pull nomic-embed-text
```

If Ollama is not already running, start `ollama serve` in a separate terminal.
Model inference and guest retrieval run locally without cloud inference keys.
Dependency installation, initial downloads, and optional web search need internet
access. Communication with Ollama uses its local HTTP interface.

## Setup and run

Create a separate virtual environment inside each project you want to use.
For example, from the repository root:

```bash
cd 02-agentic-rag-using-llamaindex
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

For the first agent, use `01-first-agent-using-smolagents` instead.
Invoke each project's `.venv/bin/python` so it uses the correct dependencies.
The commands above use the macOS/Linux virtual environment layout.

**To search the internet in Alfred's chat, check “Enable web search (requires
internet)” above the chat before submitting your request.** Web search is
disabled by default. You can also start with it enabled:

```bash
.venv/bin/python app.py --enable-web
```

Alfred's weather tool returns explicitly simulated course data, not a live
forecast. Its guestbook contains three fictional event profiles from the course
dataset. Refer to each project's README for tool behavior, examples, and limitations.

## VS Code

Open [agents-course.code-workspace](agents-course.code-workspace) through
**File > Open Workspace from File**. Each project is a separate workspace folder
with its own Python environment setting.

Open the desired project's `app.py` and click **Run Python File**, or choose a
project configuration from **Run and Debug**. For Alfred, the **with web search**
configuration enables the web tool at startup. To use the integrated terminal,
right-click a project folder and select **Open in Integrated Terminal**, then run
`.venv/bin/python app.py`.

## Checks and diagnostics

From either project directory:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
```

Alfred also logs its active tools, model steps, code execution, web queries,
returned source URLs, and errors to the console. For more detail:

```bash
.venv/bin/python app.py --enable-web --log-level DEBUG
```

## Course references

- [Course introduction and overview](https://huggingface.co/learn/agents-course/unit0/introduction)
- [Unit 1: First agent with smolagents](https://huggingface.co/learn/agents-course/unit1/tutorial)
- [Unit 3: Agentic RAG](https://huggingface.co/learn/agents-course/unit3/agentic-rag/agentic-rag)
