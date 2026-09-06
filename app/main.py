"""
Application entrypoint.

This file's only job is to construct the FastAPI app and register
routers. It intentionally contains no business logic and no route
handlers of its own — those live in `app/api/*`. As the project grows
(ingestion, retrieval, financial, features routers), this file gains
one `app.include_router(...)` line per module and nothing else. That
is the entire scaling story for this file.
"""

from fastapi import FastAPI

from app.api import health
from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Evidence-driven financial due-diligence RAG platform.",
    version="0.1.0",
)

app.include_router(health.router)
