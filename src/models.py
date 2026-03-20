"""
SQLAlchemy models for fin_rag (PostgreSQL).
Used by database.py and Alembic; all non-vector DB access in src/ goes through these models.
"""
from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all models."""

    pass


class Feedback(Base):
    """User feedback and ratings for Q&A responses (was feedback.db)."""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    response_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    user: Mapped[str | None] = mapped_column(String(255), nullable=True)
    feedback: Mapped[str | None] = mapped_column(Text, nullable=True)
    question: Mapped[str | None] = mapped_column(Text, nullable=True)
    response: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_rag: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    log: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class FrequentQAPair(Base):
    """Frequent Q&A pairs (was frequent_qa.db frequent_qa_pairs)."""

    __tablename__ = "frequent_qa_pairs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    question_rewritten: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    category: Mapped[str] = mapped_column(String(255), nullable=False)
    last_updated: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    view_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    tags: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[str | None] = mapped_column("metadata", Text, nullable=True)


class QATable(Base):
    """QA table with fixed columns + period data as JSONB (was qa_table.db qa_table)."""

    __tablename__ = "qa_table"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    question_rewritten: Mapped[str] = mapped_column(Text, nullable=False)
    last_updated: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    period_data: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)


class AnswerHistory(Base):
    """History of answer updates for frequent QA pairs (from script create_frequentQA_database)."""

    __tablename__ = "answer_history"

    history_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    qa_id: Mapped[int | None] = mapped_column(ForeignKey("frequent_qa_pairs.id"), nullable=True)
    previous_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    version: Mapped[int | None] = mapped_column(Integer, nullable=True)


class FeedbackQuestionAlias(Base):
    """Processed feedback records: question aliases and categorization (from script frequentQA_db_processor)."""

    __tablename__ = "feedback_question_aliases"

    alias_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    qa_id: Mapped[int | None] = mapped_column(ForeignKey("frequent_qa_pairs.id"), nullable=True)
    alias_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    response_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    rating: Mapped[int | None] = mapped_column(Integer, nullable=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    question_rewritten: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(255), nullable=False, server_default=text("'non_rag'"))
    is_match: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    match_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, server_default=text("now()"), nullable=True)
    approved_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
