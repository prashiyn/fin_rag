"""add answer_history and feedback_question_aliases

Revision ID: 002
Revises: 001
Create Date: 2025-02-22

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "answer_history",
        sa.Column("history_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("qa_id", sa.Integer(), nullable=True),
        sa.Column("previous_answer", sa.Text(), nullable=True),
        sa.Column("updated_answer", sa.Text(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.Column("updated_by", sa.String(length=255), nullable=True),
        sa.Column("version", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["qa_id"], ["frequent_qa_pairs.id"]),
        sa.PrimaryKeyConstraint("history_id"),
    )
    op.create_index(op.f("ix_answer_history_qa_id"), "answer_history", ["qa_id"], unique=False)

    op.create_table(
        "feedback_question_aliases",
        sa.Column("alias_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("qa_id", sa.Integer(), nullable=True),
        sa.Column("alias_text", sa.Text(), nullable=True),
        sa.Column("session_id", sa.String(length=255), nullable=True),
        sa.Column("response_id", sa.String(length=255), nullable=True),
        sa.Column("rating", sa.Integer(), nullable=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("question_rewritten", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("category", sa.String(length=255), nullable=False, server_default=sa.text("'non_rag'")),
        sa.Column("is_match", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("match_confidence", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=True),
        sa.Column("approved_by", sa.String(length=255), nullable=True),
        sa.Column("approved_at", sa.DateTime(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["qa_id"], ["frequent_qa_pairs.id"]),
        sa.PrimaryKeyConstraint("alias_id"),
    )
    op.create_index(op.f("ix_feedback_question_aliases_qa_id"), "feedback_question_aliases", ["qa_id"], unique=False)
    op.create_index(op.f("ix_feedback_question_aliases_response_id"), "feedback_question_aliases", ["response_id"], unique=False)
    op.create_index(op.f("ix_feedback_question_aliases_session_id"), "feedback_question_aliases", ["session_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_feedback_question_aliases_session_id"), table_name="feedback_question_aliases")
    op.drop_index(op.f("ix_feedback_question_aliases_response_id"), table_name="feedback_question_aliases")
    op.drop_index(op.f("ix_feedback_question_aliases_qa_id"), table_name="feedback_question_aliases")
    op.drop_table("feedback_question_aliases")
    op.drop_index(op.f("ix_answer_history_qa_id"), table_name="answer_history")
    op.drop_table("answer_history")
