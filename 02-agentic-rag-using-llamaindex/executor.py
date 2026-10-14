"""Execute CodeAct snippets with prepared tools in a restricted AST interpreter."""

import ast
from functools import wraps
import logging
from time import perf_counter

from diagnostics import preview, request_id

logger = logging.getLogger("alfred.executor")

from smolagents.local_python_executor import LocalPythonExecutor


class CodeExecutor:
    """One task's persistent variables and tool calls; not an OS security sandbox."""

    def __init__(self, functions):
        self.request_id = request_id.get()
        self.tool_names = {function.__name__ for function in functions}
        self.calls = []
        self.interpreter = LocalPythonExecutor(
            additional_authorized_imports=[], max_print_outputs_length=12000, timeout_seconds=120
        )
        self.interpreter.send_tools({function.__name__: self._record(function) for function in functions})

    def _record(self, function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            token = request_id.set(self.request_id)
            try:
                return self._invoke(function, args, kwargs)
            finally:
                request_id.reset(token)
        return wrapped

    def _invoke(self, function, args, kwargs):
        started = perf_counter()
        logger.info("TOOL START name=%s args=%s kwargs=%s", function.__name__, preview(args, 500), preview(kwargs, 500))
        call = {"tool": function.__name__, "args": list(args), "kwargs": kwargs}
        self.calls.append(call)
        try:
            output = function(*args, **kwargs)
        except Exception as error:
            logger.error("TOOL FAILED name=%s duration=%.2fs error=%s", function.__name__, perf_counter() - started, error)
            call["error"] = str(error)
            raise
        call["result"] = output
        if isinstance(output, dict) and output.get("error"):
            logger.warning("TOOL ERROR RESULT name=%s error=%s", function.__name__, output["error"])
        logger.info("TOOL END name=%s duration=%.2fs", function.__name__, perf_counter() - started)
        logger.debug("TOOL RESULT name=%s result=%s", function.__name__, preview(output, 8000))
        return output

    def execute(self, code: str) -> dict:
        """Execute a Python snippet and return printed observations or a clear error."""
        token = request_id.set(self.request_id)
        try:
            return self._execute(code)
        finally:
            request_id.reset(token)

    def _execute(self, code: str) -> dict:
        start = len(self.calls)
        try:
            if len(code) > 20000:
                raise ValueError("Code snippet is too large.")
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    raise ValueError("Imports are disabled. Call the prepared tools directly.")
                if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
                    raise ValueError("Private attributes are disabled.")
                if isinstance(node, ast.Name) and node.id.startswith("__"):
                    raise ValueError("Private names are disabled.")
                if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and node.id in self.tool_names:
                    raise ValueError("Do not overwrite prepared tools.")
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in self.tool_names:
                    raise ValueError("Do not redefine prepared tools.")
            result = self.interpreter(code)
            return {"stdout": result.logs, "result": result.output, "tool_calls": self.calls[start:]}
        except Exception as error:
            logger.error("PYTHON EXECUTION FAILED %s: %s", type(error).__name__, error)
            return {"error": f"{type(error).__name__}: {error}", "tool_calls": self.calls[start:]}
