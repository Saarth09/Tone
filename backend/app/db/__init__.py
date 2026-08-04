from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings
from app.db.models import Base

settings = get_settings()


def normalize_database_url(url: str) -> str:
    """Accept common Postgres URLs and force the async driver."""
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    if url.startswith("postgresql://") and "+asyncpg" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


DATABASE_URL = normalize_database_url(settings.database_url)

if DATABASE_URL.startswith("sqlite"):
    db_path = DATABASE_URL.split("///")[-1]
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def init_db() -> None:
    """Create all application tables (users, samples, drift, alerts, connections)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        if DATABASE_URL.startswith("sqlite"):
            await _sqlite_add_column_if_missing(conn, "samples", "user_id", "INTEGER")
            await _sqlite_add_column_if_missing(conn, "drift_scores", "user_id", "INTEGER")
            await _sqlite_add_column_if_missing(conn, "alert_events", "user_id", "INTEGER")
            await _sqlite_add_column_if_missing(conn, "llm_connections", "user_id", "INTEGER")
            await _sqlite_add_column_if_missing(
                conn, "users", "auth_provider", "VARCHAR(32) DEFAULT 'password'"
            )
            await _sqlite_add_column_if_missing(conn, "users", "google_sub", "VARCHAR(128)")
            await _sqlite_add_column_if_missing(conn, "users", "avatar_url", "VARCHAR(512)")


async def _sqlite_add_column_if_missing(conn, table: str, column: str, coltype: str) -> None:
    result = await conn.execute(text(f"PRAGMA table_info({table})"))
    cols = {row[1] for row in result.fetchall()}
    if column not in cols:
        await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}"))


async def get_session():
    async with SessionLocal() as session:
        yield session


async def database_info() -> dict:
    dialect = engine.dialect.name
    async with engine.connect() as conn:
        if dialect == "postgresql":
            version = (await conn.execute(text("SELECT version()"))).scalar()
            db_name = (await conn.execute(text("SELECT current_database()"))).scalar()
        else:
            version = "sqlite"
            db_name = DATABASE_URL
    return {
        "dialect": dialect,
        "database": db_name,
        "version": version,
        "url_scheme": DATABASE_URL.split("://", 1)[0],
    }
