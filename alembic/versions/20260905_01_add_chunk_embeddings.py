"""Add pgvector embeddings to document chunks.

Revision ID: 20260905_01
Revises: 20260905_00
Create Date: 2026-09-05
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = "20260905_01"
down_revision: Union[str, Sequence[str], None] = "20260905_00"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.add_column("chunks", sa.Column("embedding", Vector(1536), nullable=True))


def downgrade() -> None:
    op.drop_column("chunks", "embedding")
