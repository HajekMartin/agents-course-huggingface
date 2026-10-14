"""Console diagnostics shared by startup, tasks, execution, and web search."""

import logging
import sys
from contextvars import ContextVar

request_id = ContextVar("alfred_request_id", default="-")


class RequestFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id.get()
        return True


def configure_logging(level="INFO"):
    logger = logging.getLogger("alfred")
    logger.setLevel(level)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.addFilter(RequestFilter())
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)


def preview(value, limit=2000):
    text = str(value)
    return text if len(text) <= limit else text[:limit] + " … [truncated]"
