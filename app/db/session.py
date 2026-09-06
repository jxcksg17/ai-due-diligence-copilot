"""
Database engine and session management.

This module knows exactly one thing: how to turn a connection string
into a SQLAlchemy engine and hand out sessions. It has no knowledge of
FastAPI, routes, or any business/domain logic, and nothing in here
imports from `app.api` or feature modules.

Why this separation matters: ingestion, retrieval, financial
calculations, and evaluation (added in later milestones) will all need
a database session, but none of them should need to know *how* that
session is constructed or *where* the connection string comes from.
They just import `get_db` and use it. If we ever change how we connect
to Postgres (connection pooling strategy, a different driver, a read
replica), this is the only file that changes.
"""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

settings = get_settings()

# `pool_pre_ping=True` makes the pool check that a connection is still
# alive before handing it out, which avoids a class of hard-to-debug
# "server closed the connection unexpectedly" errors after idle periods.
engine = create_engine(settings.database_url, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session per-request.

    The `yield`/`finally` pattern guarantees the session is closed even
    if the request handler raises an exception, preventing connection
    leaks under error conditions.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
