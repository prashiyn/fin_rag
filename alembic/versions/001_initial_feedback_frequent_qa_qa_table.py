"""initial feedback, frequent_qa_pairs, qa_table

Revision ID: 001
Revises:
Create Date: 2025-02-22

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "feedback",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(length=255), nullable=False),
        sa.Column("response_id", sa.String(length=255), nullable=False),
        sa.Column("rating", sa.Integer(), nullable=False),
        sa.Column("user", sa.String(length=255), nullable=True),
        sa.Column("feedback", sa.Text(), nullable=True),
        sa.Column("question", sa.Text(), nullable=True),
        sa.Column("response", sa.Text(), nullable=True),
        sa.Column("is_rag", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("log", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_feedback_response_id"), "feedback", ["response_id"], unique=False)
    op.create_index(op.f("ix_feedback_session_id"), "feedback", ["session_id"], unique=False)

    op.create_table(
        "frequent_qa_pairs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("question_rewritten", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("category", sa.String(length=255), nullable=False),
        sa.Column("last_updated", sa.DateTime(), nullable=True),
        sa.Column("updated_by", sa.String(length=255), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("view_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("version", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("tags", sa.Text(), nullable=True),
        sa.Column("metadata", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_frequent_qa_pairs_category"), "frequent_qa_pairs", ["category"], unique=False)

    op.create_table(
        "qa_table",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("question_rewritten", sa.Text(), nullable=False),
        sa.Column("last_updated", sa.DateTime(), nullable=True),
        sa.Column("updated_by", sa.String(length=255), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("period_data", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("qa_table")
    op.drop_index(op.f("ix_frequent_qa_pairs_category"), table_name="frequent_qa_pairs")
    op.drop_table("frequent_qa_pairs")
    op.drop_index(op.f("ix_feedback_session_id"), table_name="feedback")
    op.drop_index(op.f("ix_feedback_response_id"), table_name="feedback")
    op.drop_table("feedback")
