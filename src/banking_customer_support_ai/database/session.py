from sqlalchemy import Engine, create_engine
from sqlalchemy.pool import StaticPool

from banking_customer_support_ai.database.models import Base


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        kwargs: dict = {"connect_args": {"check_same_thread": False}}
        if url in ("sqlite://", "sqlite:///:memory:"):
            kwargs["poolclass"] = StaticPool  # one shared in-memory DB across threads
        return create_engine(url, **kwargs)
    return create_engine(url, pool_pre_ping=True)


def init_db(engine: Engine) -> None:
    """Create tables if missing. Idempotent."""
    Base.metadata.create_all(engine)
