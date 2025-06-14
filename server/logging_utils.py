import contextvars
from logging import Filter, LogRecord

# Context variable storing the current request ID
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

class RequestIdFilter(Filter):
    """Logging filter to inject the request ID into log records."""

    def filter(self, record: LogRecord) -> bool:
        record.request_id = request_id_var.get("-")
        return True
