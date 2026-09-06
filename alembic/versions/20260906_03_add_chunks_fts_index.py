"""Add PostgreSQL full-text search index for chunk text.

Revision ID: 20260906_03
Revises: 20260905_02
"""

from collections.abc import Sequence

from alembic import op


revision: str = "20260906_03"
down_revision: str | None = "20260905_02"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX ix_chunks_text_fts ON chunks "
        "USING gin (to_tsvector('english'::regconfig, text))"
    )


def downgrade() -> None:
    op.execute("DROP INDEX ix_chunks_text_fts")
