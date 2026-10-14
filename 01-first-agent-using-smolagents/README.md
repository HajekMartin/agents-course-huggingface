# First Agent with Local Ollama

A local adaptation of the Hugging Face Agents Course, Unit 1:
[Let's Create Our First Agent Using smolagents](https://huggingface.co/learn/agents-course/unit1/tutorial).
The original [First Agent template](https://huggingface.co/spaces/agents-course/First_agent_template/tree/main)
is the reference for this project.

The application keeps the tutorial's `CodeAgent`, typed `@tool` functions,
`FinalAnswerTool`, six-step agent loop, YAML instructions, and Gradio chat UI.
`LiteLLMModel` connects to **your local Ollama** instead of Hugging Face's
`InferenceClientModel`. No Hugging Face token, OpenAI key, paid service, or
cloud inference is required. Communication with Ollama uses its local HTTP
interface at `http://localhost:11434`; this is not a cloud API.

The dummy tool is replaced with `roll_dice`: it chooses a **random number of
rolls**, then generates that many independent six-sided die results. The time
zone tool also runs locally. Web search and hosted image generation are omitted
to keep the agent local. Internet access is needed to install dependencies and
download an Ollama model initially.

## Requirements

- Python 3.10 or newer.
- [Ollama](https://ollama.com/) installed and running locally.
- A locally downloaded model. The default is `qwen2:7b`, matching the model
  installed when this project was created. You can select another installed model.

## Setup and run

From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If Ollama is already running, leave it running. Otherwise start it in a separate
terminal:

```bash
ollama serve
```

Check your models and download the default only if it is missing:

```bash
ollama list
ollama pull qwen2:7b
```

Start the chat UI:

```bash
python app.py
```

Open **http://127.0.0.1:7860**. The UI binds to your computer and does not create
a public Gradio share link.

For a single task in the terminal:

```bash
python app.py --prompt "Roll a random number of dice between 2 and 10 times. Show every result and the total."
```

## Configuration

### Run from VS Code

For automatic interpreter selection between lessons, open the parent
`agents-course.code-workspace` file through **File > Open Workspace from File**.
Each lesson is a separate workspace folder with its own `.venv` setting. Open
this lesson's `app.py` and click **Run Python File**. Switching to another lesson's
file uses that folder's Python environment. Each folder also has explicit launch
configurations available in **Run and Debug**.

Open the parent `agents-course` directory in VS Code. The parent `.vscode/launch.json`
defines **01 - First Agent (Ollama)** with an explicit `app.py`, working directory,
and Python interpreter in this project's `.venv`.

Install the recommended Microsoft Python and Python Debugger extensions, open
**Run and Debug**, select **01 - First Agent (Ollama)**, then press **F5** to debug
or choose **Run > Run Without Debugging** to run. This configuration starts
this app regardless of the file currently open in the editor.

The editor's **Run Python File** button runs the active Python file. For that
button, select this project's `.venv/bin/python` through **Python: Select Interpreter**
if VS Code still remembers an older interpreter. The parent workspace settings
also make this button run from the active file's directory.

If you move a virtual environment, its activation scripts may retain the old
absolute path. This launch configuration invokes the interpreter directly and
does not require `source .venv/bin/activate`.

### Model and host

Choose any model already shown by `ollama list`:

```bash
python app.py --model qwen2:7b
python app.py --ollama-host http://127.0.0.1:11434 --port 7861
```

`OLLAMA_MODEL` and `OLLAMA_HOST` environment variables set the defaults;
command-line arguments override them. `OLLAMA_HOST` must be a full HTTP(S)
URL using `localhost`, `127.0.0.1`, or `[::1]`. Use a locally downloaded model,
not an Ollama cloud model.

```bash
export OLLAMA_MODEL=qwen2:7b
export OLLAMA_HOST=http://localhost:11434
python app.py
```

Inference uses temperature `0.2`, a 2,096-token output limit, a 16,384-token
context, and a 180-second request timeout. These settings are in `create_agent`
in `app.py`. Larger contexts consume more memory; adjust `num_ctx` if needed.

## Tools and example prompts

### Random dice rolls

`roll_dice(min_rolls=1, max_rolls=6)` chooses the count uniformly from the
inclusive bounds, then independently chooses each die result from 1 through 6.
Bounds must be integers satisfying `1 <= min_rolls <= max_rolls <= 1000`.
Its dictionary result includes `count`, `rolls`, and `total`, shown as JSON here:

```json
{"count": 3, "rolls": [2, 6, 1], "total": 9}
```

This is an illustrative result; subsequent calls can produce different counts
and values. Equal bounds specify an exact count, such as `(5, 5)` for five rolls.

Try:

- "Roll a random number of dice and show the results."
- "Roll between 3 and 12 times. What is the total?"
- "Roll exactly 5 times and tell me how many sixes you got."

### Local time

`get_local_timezone()` reads the timezone configured on the computer running
the app using [tzlocal](https://github.com/regebro/tzlocal). It returns the
timezone name, current UTC offset, abbreviation, and local time. The offset
reflects daylight saving time: `Europe/Prague` uses `+01:00` in winter and
`+02:00` in summer. The timezone is detected from system settings, not guessed
from language or location. If the UI is accessed remotely, this result still
describes the computer running the app, not the browser's timezone.

Try: "What is my current timezone?" or "What is the current local time?"

`get_current_time_in_timezone(timezone)` uses Python's `zoneinfo` and your
computer's clock, including the timezone offset and daylight saving rules.

- "What time is it in Europe/Prague?"
- "Roll some dice, then tell me the current time in America/New_York."

## Project structure

```text
app.py             # Tools, Ollama model, CodeAgent, CLI, and Gradio launch
Gradio_UI.py       # Local UI using smolagents' streaming chat and agent trace
prompts.yaml       # Shared agent rules, general examples, and the tool rendering loop
requirements.txt   # Python dependencies
tests/test_app.py  # Tool and agent-loop checks without an LLM
```

Like the course template, `prompts.yaml` supplies the agent's system prompt.
Tool-specific usage rules and task examples live in each tool function's docstring
in `app.py`. General examples, such as greetings and unavailable capabilities,
stay in `prompts.yaml`.
The `@tool` decorator reads these descriptions, and the YAML loop inserts
them automatically through `tool.to_code_prompt()`. When adding a tool, document
its usage and examples beside its implementation and register it in the agent's
`tools` list. Tool examples use the default CodeAgent `<code>` tags; if you change
the agent's code delimiters, update these examples too.
General rules for observed results, errors, and completion stay in the YAML.
The rendered prompt contains tool definitions and short dice/time examples suited to a local
model. The remaining planning and final-answer prompt templates are loaded
from the installed smolagents version and merged with this system prompt.
`Gradio_UI.py` reuses smolagents' streaming response handler and agent trace,
with a local chat layout that avoids the default UI's remotely hosted avatar.
The final-answer tool comes directly from smolagents.
Each submitted message starts a fresh agent run so failed actions from earlier
tasks do not pollute subsequent prompts. The visible chat history is retained,
but follow-up requests must include their own task details.

## Verification

```bash
python -m unittest discover -s tests -v
python -m pip check
```

The tests run without Ollama or external services. They check dice boundaries,
invalid inputs, timezone handling, startup errors, and an agent actually executing
the dice tool before returning its observed result. To verify your local model,
run the terminal example above and look for a `roll_dice` execution in the trace.

## Troubleshooting

- **Cannot reach Ollama:** check `ollama serve` and the configured host.
- **Model is not installed:** use `ollama list`, select it with `--model`, or
  download it with `ollama pull`.
- **Port is in use:** start with `python app.py --port 7861`.
- **Slow responses or code parsing errors:** local models vary in their ability
  to follow the agent's Python code format. Try a larger or code-oriented local
  model; do not assume the agent used a tool unless its execution appears in the trace.
- **Agent repeats a final answer:** `CodeAgent` requires a call to
  `final_answer(...)` inside a `<code>` block, including for greetings and
  unsupported requests. The system prompt includes examples of these cases.
  Restart the app after modifying the prompt. Live weather is unavailable with
  the tools in this project.

## References

- [Hugging Face Agents Course tutorial](https://huggingface.co/learn/agents-course/unit1/tutorial)
- [Original First Agent template](https://huggingface.co/spaces/agents-course/First_agent_template/tree/main)
- [smolagents model documentation](https://huggingface.co/docs/smolagents/reference/models)
- [LiteLLM Ollama integration](https://docs.litellm.ai/docs/providers/ollama)
- [Ollama local interface documentation](https://docs.ollama.com/api/introduction)
