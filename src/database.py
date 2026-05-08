"""
PostgreSQL session and engine for src/ (feedback, frequent_qa_pairs, qa_table).
All DB access in src/ uses get_session() or the session factory here.
"""
import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from models import Base

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal: sessionmaker[Session] | None = None


def init_db(
    database_url: str,
    *,
    pool_size: int = 5,
    max_overflow: int = 10,
) -> None:
    """Create engine and session factory. Call once at app startup (e.g. from config)."""
    global _engine, _SessionLocal
    if _engine is not None:
        return
    pool_size = max(1, int(pool_size))
    max_overflow = max(0, int(max_overflow))
    _engine = create_engine(
        database_url,
        pool_pre_ping=True,
        pool_size=pool_size,
        max_overflow=max_overflow,
    )
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    logger.info("Database engine and session factory initialized")


def get_engine():
    """Return the global engine. init_db must have been called."""
    if _engine is None:
        raise RuntimeError("Database not initialized; call init_db(database_url) first")
    return _engine


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Yield a session for the current request/operation. Commit on exit, rollback on exception."""
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized; call init_db(database_url) first")
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def close_db() -> None:
    """Dispose engine (e.g. on app shutdown)."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
        _engine = None
        _SessionLocal = None
    logger.info("Database engine disposed")
