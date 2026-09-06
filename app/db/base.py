"""
Shared declarative base.

Kept in its own tiny module (rather than inside models.py or
session.py) so that both `app/db/models.py` and any future migration
tooling can import `Base` without creating circular imports between
"the engine/session" and "the table definitions."
"""

from sqlalchemy.orm import declarative_base

Base = declarative_base()
