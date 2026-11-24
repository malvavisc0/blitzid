"""Database engine and session wiring for the hosted website."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from fastapi import Request
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    """Declarative base for every hosted-SaaS table."""


class Database:
    """Engine + session factory, built once per app."""

    def __init__(self, url: str) -> None:
        options: dict[str, Any] = {"pool_pre_ping": True}
        if url.startswith("sqlite"):
            options["connect_args"] = {"check_same_thread": False}
            if ":memory:" in url:
                options["poolclass"] = StaticPool
        self.engine = create_engine(url, **options)
        self.session_factory = sessionmaker(
            bind=self.engine, expire_on_commit=False, autoflush=False
        )

    def create_all(self) -> None:
        """Create the schema directly (tests and first-run local dev)."""
        Base.metadata.create_all(self.engine)


def get_db(request: Request) -> Iterator[Session]:
    """FastAPI dependency: one session per request, closed afterwards."""
    database: Database = request.app.state.db
    with database.session_factory() as session:
        yield session
