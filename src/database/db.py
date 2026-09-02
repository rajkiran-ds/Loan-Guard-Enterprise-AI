"""
src/database/db.py
--------------------
Engine/session lifecycle + a small repository layer on top of models.Application.
Streamlit pages and the decision pipeline should only ever call functions in
this file — never touch SQLAlchemy sessions directly — so the persistence
layer can be swapped (e.g. SQLite -> Postgres) without touching app.py.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config import DATABASE_PATH
from src.database.models import Application, Base
from src.utils.logger import get_logger

logger = get_logger(__name__)

_engine = create_engine(f"sqlite:///{DATABASE_PATH}", echo=False, future=True)
SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, future=True)


def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every app start."""
    Base.metadata.create_all(_engine)
    logger.info("Database initialized at %s", DATABASE_PATH)


@contextmanager
def get_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("Database transaction failed — rolled back.")
        raise
    finally:
        session.close()


def save_application(record: dict) -> int:
    """Persist one full loan application record. Returns the new row id."""
    with get_session() as session:
        application = Application(**record)
        session.add(application)
        session.flush()
        app_id = application.id
        logger.info("Saved application id=%s decision=%s", app_id, record.get("decision"))
        return app_id


def get_application(app_id: int) -> Application | None:
    with get_session() as session:
        return session.get(Application, app_id)


def list_applications_df() -> pd.DataFrame:
    """Return all applications as a DataFrame for the executive dashboard."""
    with get_session() as session:
        rows = session.query(Application).all()
        records = [
            {c.name: getattr(row, c.name) for c in Application.__table__.columns}
            for row in rows
        ]
    return pd.DataFrame(records)


def application_count() -> int:
    with get_session() as session:
        return session.query(Application).count()
