"""
Alembic env: uses config/production.yaml (or CONFIG_PATH) database_url for migrations.
DATABASE_URL env (e.g. from .env) overrides config. Autogenerate: alembic revision --autogenerate -m "msg"
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root so DATABASE_URL / CONFIG_PATH are set
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

import yaml
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context

# Ensure src is on path so "from models import Base" works
_src = _project_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from models import Base

config = context.config
if config.config_file_name is not None:
    from logging.config import fileConfig
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_database_url() -> str:
    """Read database_url from config/production.yaml or CONFIG_PATH, else DATABASE_URL env."""
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    config_path = os.environ.get("CONFIG_PATH")
    if not config_path:
        config_path = _project_root / "config" / "production.yaml"
    else:
        config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        url = data.get("database_url")
        if url:
            return url
    # Fallback from alembic.ini (user must set it or use env)
    return config.get_main_option("sqlalchemy.url", "postgresql://postgres:postgres@localhost:5432/fin_rag")


def run_migrations_offline() -> None:
    url = get_database_url()
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
    configuration["sqlalchemy.url"] = get_database_url()
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
