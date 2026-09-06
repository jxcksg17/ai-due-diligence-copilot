"""Use BGE's native 1024-dimensional embedding vectors.

Revision ID: 20260905_02
Revises: 20260905_01
Create Date: 2026-09-05
"""

from typing import Sequence, Union

from alembic import op
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = "20260905_02"
down_revision: Union[str, Sequence[str], None] = "20260905_01"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _require_no_embeddings() -> None:
    """Prevent a type change from silently discarding incompatible vectors."""
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM chunks WHERE embedding IS NOT NULL) THEN
                RAISE EXCEPTION
                    'Cannot change embedding dimensions while embeddings exist';
            END IF;
        END $$;
        """
    )


def upgrade() -> None:
    _require_no_embeddings()
    op.alter_column(
        "chunks",
        "embedding",
        existing_type=Vector(1536),
        type_=Vector(1024),
        postgresql_using="embedding::vector(1024)",
    )


def downgrade() -> None:
    _require_no_embeddings()
    op.alter_column(
        "chunks",
        "embedding",
        existing_type=Vector(1024),
        type_=Vector(1536),
        postgresql_using="embedding::vector(1536)",
    )
