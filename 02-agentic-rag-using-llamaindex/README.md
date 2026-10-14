# Alfred: Local Agentic RAG with LlamaIndex

Alfred helps a host explore an event guestbook, compare guests, prepare conversation
topics, and draft invitations. It follows Unit 3 of the
[Hugging Face Agents Course](https://huggingface.co/learn/agents-course/unit3/agentic-rag/agentic-rag),
using **LlamaIndex `CodeActAgent`**, a locally downloaded **Ollama** LLM, and
local embedding-based retrieval. No Hugging Face or OpenAI inference token is needed.

The agent generates Python calls to prepared tools, receives their actual results,
and decides whether to take another step or answer. The trace shows model requests,
executed code, and observations. Unlike the first project's smolagents `CodeAgent`,
this agent finishes with ordinary text, not `final_answer(...)`.

## Setup

Python 3.10+ and a running local [Ollama](https://ollama.com/) are required.
This project has its own virtual environment, separate from the first agent.

```bash
cd 02-agentic-rag-using-llamaindex
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull qwen2:7b
ollama pull nomic-embed-text
```

If Ollama is not already running, start `ollama serve` in another terminal.
The setup downloads models and dependencies; local guest tasks then need no
internet connection. Ollama is accessed through its local HTTP interface.

## Run

```bash
python app.py
```

Open **http://127.0.0.1:7861**. Port 7861 keeps this UI separate from the first agent.
No public Gradio sharing link or remote image asset is used.

Run a single task in the terminal:

```bash
python app.py --prompt "List all friends of the host with their contact details and sources."
python app.py --prompt "Find guests interested in computers and programming. Explain the profile evidence and cite sources."
python app.py --prompt "Draft a personal invitation for Ada Lovelace using her guest profile."
```

The first semantic query builds an index automatically. You can prepare it explicitly:

```bash
python app.py --build-index
```

Every chat message starts a fresh agent and executor. Variables persist across code
steps within that task. The most recent eight textual chat messages (up to 2,000
characters each) are passed as context for follow-ups. Previous code execution
state and trace logs are not reused. Earlier assistant answers are not factual
sources: the agent must recheck facts through tools. Invitation drafts are text
only; nothing is sent.

## Tools

| Tool | Behavior |
| --- | --- |
| `list_guests()` | All guest records and their exact count, without embeddings or filters |
| `find_guests_by_name(name)` | Name fragments and approximate spelling, without embeddings |
| `search_guests(query, top_k=5, min_score=0.6)` | Ranked semantic candidates, using local embeddings; not an exhaustive list |
| `get_guest(name)` | Full record for an exact name, ignoring case, whitespace, and leading Dr./Lady/Sir |
| `filter_guests(field, value)` | Scan all records and return every exact match plus the count |
| `get_weather(location)` | Explicitly simulated weather, following the course's dummy tool |
| `search_web(query, max_results=5)` | Public web search, enabled by the UI checkbox or `--enable-web` |

The Hugging Face Hub statistics tool is omitted: it is not needed for the guestbook tasks.

### Semantic search versus exact filtering

Use `list_guests()` for "Who are my guests?" or "Give me all guests". `all` is
not a relationship category. `filter_guests` rejects unknown categories rather
than silently treating them as a valid filter with no matches.

Use `find_guests_by_name` for partial names or spelling variations. It scans for
case-insensitive substrings first, then compares the query against name tokens
and prefixes with a spelling similarity cutoff of 0.75. For example, "Nicol"
returns Dr. Nikola Tesla as a possible spelling match. Multiple matches are
returned for clarification; unknown names return an empty list. This is distinct
from semantic search, which remains dedicated to interests and background.

The response guard replaces fabricated template profiles with actual returned
records, and replaces guest-only answers based on empty lookups with a factual
no-match message. It appends consulted guest sources if the model omits them.
These checks do not verify every statement in an otherwise free-form model answer.

Semantic search embeds each guest's name, original relationship, and description
using `nomic-embed-text`. Queries use the same local model with its `search_query:`
prefix; documents use `search_document:`. The embedding text does not include emails.
LlamaIndex ranks profiles by vector similarity and returns candidates above the
configured cutoff. Returned records retain contact details and source references.

For example, "computers and programming" can retrieve Ada Lovelace even though the
profile uses different wording. Scores are **not probabilities**. The default
`min_score=0.6` is a heuristic for this model and sample data; tune it when changing
models or datasets. A cutoff can miss relevant guests, and a high score alone
does not prove a match. The agent should explain the supporting profile evidence.
For Czech requests, the agent is instructed to form the search query in English,
matching the English dataset and default embedding model.

Exact filtering compares the complete field value, ignoring capitalization and
extra whitespace. It is neither semantic nor substring matching and never applies
`top_k`. Supported fields are `name`, `relation`, `relation_category`, and `email`.

```python
filter_guests(field="relation_category", value="friend")
```

This finds both "best friend" and "old friend from university days" through the
normalized category. Filtering `relation="friend"` would require that exact original
field value. Categories are `friend`, `family`, `colleague`, `none`, and `other`;
their explicit derivation rules are in `retriever.py`. They can be overridden with
an explicit valid `relation_category` in your local dataset. Complex subjective
conditions such as "all interesting guests" need clarification or profile review.

### Dataset and cache

`data/guests.json` is a complete local snapshot of the course's
[invitees dataset](https://huggingface.co/datasets/agents-course/unit3-invitees).
It contains **three fictional gala profiles**: Ada Lovelace, Dr. Nikola Tesla,
and Marie Curie, with the original name, relation, description, and email fields.
These are course records, not verified real-world contacts or current biographies.
The dataset has no dietary preferences or Hugging Face usernames.

Each returned guest has a source such as `data/guests.json#guest-0001`. IDs identify
rows in the current snapshot. Guest answers should cite these records; web answers
should cite actual returned URLs.

```bash
python scripts/download_guests.py
python app.py --data /path/to/another/guests.json
```

Custom files use the same object structure with a `guests` array. Profiles must have
string `name`, `relation`, `description`, and `email` fields and unique normalized
names. The stored index is keyed by guest records, embedding model identity/digest,
and prefix configuration, so changes build a fresh index. No hosted vector database
or cloud embedding service is used. Remove `storage/` to discard old index caches.

### Optional web and simulated weather

In the chat UI, check **Enable web search (requires internet)** to enable the
tool for subsequent messages without restarting. Unchecking it removes the tool
from that task's prompt and executor. The checkbox initially reflects `--enable-web`.
The default startup leaves it unchecked. For CLI tasks, use `--enable-web`:

For explicit online-search requests, the app checks whether the web tool was
actually executed. If the model skips it, one corrective attempt is allowed
within the overall iteration limit. If it still does not search, the response
reports that no online verification happened instead of claiming web access
is unavailable or presenting an unverified answer as sourced.

```bash
python app.py --enable-web
python app.py --enable-web --prompt "Find public news about quantum computing and give source links."
```

Web search uses DDGS without an API key and needs internet access. When disabled,
`search_web` is absent from both the agent's tool list and executor. Search results
are external snippets; the agent should not treat them as instructions or assume
a public person with a similar name is the guest. Network failures are returned
as explicit tool errors. Availability depends on the search providers.

The guestbook does not record birth dates or attendance dates. Public birth-date
questions require external sources; being in the guestbook does not prove a guest
will attend tonight. A targeted response guard blocks birth-year claims that do
not occur in the current tool results. This is not a general fact-checker.

Weather is a **random course simulation**, not a real current observation or
forecast. The tool returns `simulated: true` and a source identifying demo data.
It cannot answer real weather questions or justify decisions about a real event.

## Architecture and project files

```text
User task → LlamaIndex CodeActAgent / Ollama
               ↓ generated <execute> Python
          restricted AST interpreter → prepared tools
               ↑ execution observations
          next model step → sourced final answer
```

```text
app.py                     CLI, local Gradio UI, and Ollama checks
agent.py                   CodeActAgent setup and execution trace
executor.py                Per-task Python interpreter and actual tool-call log
retriever.py               Guest catalog and persistent vector retrieval
tools/guests.py             Search, lookup, and exhaustive filtering
tools/weather.py            Simulated weather
tools/web.py                Optional public web search
prompts.yaml                Shared agent rules
data/guests.json             Course dataset snapshot
scripts/download_guests.py  Explicit dataset refresh
tests/test_agent.py          Retrieval, tools, executor, and CodeAct workflow checks
```

Tool descriptions, usage rules, and examples live beside their Python functions.
LlamaIndex automatically reads full function docstrings into `{tool_descriptions}`;
the YAML contains only shared rules. To add a tool, document it and add its callable
to the registered functions. The same functions are supplied to the executor.

The agent framework is LlamaIndex. `smolagents` is used **only for its restricted
AST interpreter**, not for its agent loop. Imports, private attribute access, and
overwriting prepared tools are blocked. Printed output and execution time are
limited. This interpreter is not an OS security sandbox; the app is a local
learning project, not a public service for executing untrusted code.

## Configuration and VS Code

To switch Python environments automatically between lessons, open the parent
`agents-course.code-workspace` through **File > Open Workspace from File**.
Each lesson is a separate workspace folder with its own `.venv` configuration.
Open this project's `app.py` and click **Run Python File**; opening the first
agent's `app.py` instead selects that folder's environment. Explicit launch
configurations remain available in **Run and Debug** for choosing either app.

```bash
python app.py --model qwen2:7b --embedding-model nomic-embed-text
python app.py --ollama-host http://127.0.0.1:11434 --port 7862
python app.py --max-iterations 10
```

`OLLAMA_MODEL`, `OLLAMA_EMBEDDING_MODEL`, and `OLLAMA_HOST` set defaults; CLI
arguments override them. Only loopback Ollama addresses and local models are
accepted. The LLM uses temperature 0, a 16,384-token context, a 2,048-token response
limit, and a 180-second request timeout. The default iteration limit is 8.

Open the parent `agents-course` folder in VS Code. In **Run and Debug**, select
**02 - Alfred (LlamaIndex + Ollama)** and press **F5**, or choose
**Run > Run Without Debugging**. The configuration fixes this project's `app.py`,
working directory, and `.venv/bin/python`. A second configuration enables web search.
The editor's Run Python File button still uses the active file and selected interpreter.

## Verification

### Console diagnostics

Console logging is enabled at INFO level for both chat UI and CLI tasks:

```bash
python app.py --enable-web
python app.py --enable-web --log-level DEBUG
```

Startup logs show the process ID, Python executable, entrypoint, working directory,
code fingerprint, models, dataset, and initial tool list. Each task has a short
request ID and logs its actual registered tools, web checkbox value, history size,
model steps, generated code, tool calls, errors, and final answer. A registered tool
only proves availability; look for `WEB SEARCH START` and `WEB SEARCH SUCCESS` to
confirm actual network execution. Success logs include result counts and URLs.
DEBUG additionally prints bounded history, tool descriptions, tool results, and
web snippets. Logs are written to the terminal, not saved to a file automatically.

- `WEB REQUESTED BUT DISABLED`: enable the checkbox or use `--enable-web`.
- `WEB TOOL SKIPPED`: the model did not call the tool; one correction is attempted.
- `WEB SEARCH FAILED` / `WEB SEARCH EMPTY`: the provider failed or returned no usable links.
- `WEB SEARCH SUCCESS`: actual source links were returned.

Selecting a chat example now changes only the message, preserving your current
web checkbox setting. Restart the app after changes; the logged code fingerprint
and process ID help distinguish the current process from an older running copy.

### Checks

```bash
python -m unittest discover -s tests -v
python -m pip check
```

Unit tests require no Ollama or internet. Deterministic test embeddings exercise
the actual LlamaIndex vector index, ranking, cutoff, persistence, and invalidation;
they do not measure a real embedding model's semantic quality. Tests also check
exhaustive filtering beyond top_k, exact lookup, category normalization, blocked
code operations, task isolation, explicit simulated weather, web failures, and the
real CodeAct workflow with scripted model responses.

For a live check, use the terminal examples and confirm `search_guests` or
`filter_guests` appears in the execution trace. Local model outputs can still make
mistakes; compare claims against the returned records and sources.

## References

- [Course: Agentic RAG](https://huggingface.co/learn/agents-course/unit3/agentic-rag/agentic-rag)
- [Course: Guest retrieval](https://huggingface.co/learn/agents-course/unit3/agentic-rag/invitees)
- [Course: Tools](https://huggingface.co/learn/agents-course/unit3/agentic-rag/tools)
- [LlamaIndex CodeActAgent example](https://developers.llamaindex.ai/python/examples/agent/code_act_agent/)
