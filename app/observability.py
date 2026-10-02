"""Concise structured logging and per-request correlation IDs."""

import json
import logging
import re
import sys
from contextvars import ContextVar
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.schemas import ErrorDetail, ErrorResponse
from app.config import Settings


_REQUEST_ID = ContextVar("request_id", default="unknown")
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def get_request_id() -> str:
    return _REQUEST_ID.get()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", get_request_id()),
        }
        fields = getattr(record, "event_fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exception"] = record.exc_info[0].__name__
        return json.dumps(payload, default=str, separators=(",", ":"))


def configure_logging(settings: Settings) -> None:
    logger = logging.getLogger("app")
    logger.setLevel(settings.log_level)
    logger.propagate = False
    if not any(getattr(handler, "_due_diligence_json", False) for handler in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        handler._due_diligence_json = True  # type: ignore[attr-defined]
        logger.addHandler(handler)


def install_request_middleware(app: FastAPI, settings: Settings) -> None:
    logger = logging.getLogger("app.requests")

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        supplied = request.headers.get("X-Request-ID", "")
        request_id = (
            supplied if _REQUEST_ID_PATTERN.fullmatch(supplied) else uuid4().hex
        )
        request.state.request_id = request_id
        token = _REQUEST_ID.set(request_id)
        started = perf_counter()
        status_code = 500
        try:
            raw_length = request.headers.get("content-length")
            if raw_length is not None:
                try:
                    content_length = int(raw_length)
                except ValueError:
                    content_length = settings.max_request_body_bytes + 1
                if content_length > settings.max_request_body_bytes:
                    response = JSONResponse(
                        status_code=413,
                        content=ErrorResponse(
                            error=ErrorDetail(
                                code="request_too_large",
                                message="request body exceeds the configured limit",
                                request_id=request_id,
                            )
                        ).model_dump(mode="json"),
                    )
                    response.headers["X-Request-ID"] = request_id
                    status_code = 413
                    return response
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            logger.info(
                "request_complete",
                extra={
                    "request_id": request_id,
                    "event_fields": {
                        "method": request.method,
                        "endpoint": request.url.path,
                        "status_code": status_code,
                        "duration_ms": round((perf_counter() - started) * 1000, 3),
                    },
                },
            )
            _REQUEST_ID.reset(token)
