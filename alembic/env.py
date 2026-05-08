"""
Alembic env: resolves DB URL via src/config.get_database_url() (POSTGRES_* or DATABASE_URL).
Autogenerate: alembic revision --autogenerate -m "msg"
"""
import sys
from pathlib import Path

from dotenv import load_dotenv

_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context

_src = _project_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from models import Base
from config import get_database_url as resolve_database_url

config = context.config
if config.config_file_name is not None:
    from logging.config import fileConfig
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def migration_database_url() -> str:
    url = resolve_database_url()
    if url and url.strip():
        return url.strip()
    return config.get_main_option("sqlalchemy.url", "postgresql://postgres:postgres@localhost:5432/finrag")


def run_migrations_offline() -> None:
    url = migration_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section, {}) or {}
    configuration["sqlalchemy.url"] = migration_database_url()
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
