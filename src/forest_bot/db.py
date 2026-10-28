"""Сховище звірів і документів у PostgreSQL (asyncpg)."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Self

import asyncpg

from forest_bot.places import Level, Scope

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    tg_id      BIGINT PRIMARY KEY,
    real_name  TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documents (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    title        TEXT NOT NULL,
    file_name    TEXT NOT NULL,
    source       TEXT NOT NULL CHECK (source IN ('archive', 'telegram')),
    owner_tg_id  BIGINT,
    owner_name   TEXT,
    level        TEXT NOT NULL CHECK (level IN ('forest', 'part', 'glove')),
    place_id     TEXT,
    summary      TEXT NOT NULL,
    content      TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    file_hash    TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_active    BOOLEAN NOT NULL DEFAULT TRUE,
    replaced_by  BIGINT REFERENCES documents (id),
    CHECK ((level = 'forest') = (place_id IS NULL))
);

CREATE INDEX IF NOT EXISTS ix_documents_scope ON documents (level, place_id) WHERE is_active;
CREATE INDEX IF NOT EXISTS ix_documents_owner ON documents (owner_tg_id) WHERE is_active;
CREATE INDEX IF NOT EXISTS ix_documents_hash ON documents (content_hash);
CREATE INDEX IF NOT EXISTS ix_documents_replaced_by ON documents (replaced_by);
"""


@dataclass(frozen=True)
class Document:
    id: int
    title: str
    file_name: str
    source: str
    owner_tg_id: int | None
    owner_name: str | None
    scope: Scope
    summary: str
    content: str
    content_hash: str
    created_at: datetime
    is_active: bool

    @property
    def from_archive(self) -> bool:
        return self.source == "archive"


@dataclass(frozen=True)
class NewDocument:
    title: str
    file_name: str
    source: str
    scope: Scope
    summary: str
    content: str
    content_hash: str
    owner_tg_id: int | None = None
    owner_name: str | None = None
    file_hash: str | None = None


def _row_to_document(row: asyncpg.Record) -> Document:
    return Document(
        id=row["id"],
        title=row["title"],
        file_name=row["file_name"],
        source=row["source"],
        owner_tg_id=row["owner_tg_id"],
        owner_name=row["owner_name"],
        scope=Scope(Level(row["level"]), row["place_id"]),
        summary=row["summary"],
        content=row["content"],
        content_hash=row["content_hash"],
        created_at=row["created_at"],
        is_active=row["is_active"],
    )


class Database:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @classmethod
    async def open(cls, dsn: str, *, schema: str | None = None) -> Self:
        """Підключається й створює таблиці. schema — окрема схема (для тестів та індексації)."""
        if schema:
            conn = await asyncpg.connect(dsn)
            try:
                await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
            finally:
                await conn.close()
        pool = await asyncpg.create_pool(
            dsn,
            min_size=1,
            max_size=5,
            server_settings={"search_path": schema} if schema else None,
        )
        # Блокування, щоб кілька процесів не створювали таблиці одночасно.
        async with pool.acquire() as pooled, pooled.transaction():
            await pooled.execute("SELECT pg_advisory_xact_lock(hashtext('forest_bot_schema'))")
            await pooled.execute(SCHEMA)
        return cls(pool)

    async def close(self) -> None:
        await self._pool.close()

    async def drop_schema(self, schema: str) -> None:
        await self._pool.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')

    # --- звірі ---------------------------------------------------------------

    async def get_user_name(self, tg_id: int) -> str | None:
        name: str | None = await self._pool.fetchval(
            "SELECT real_name FROM users WHERE tg_id = $1", tg_id
        )
        return name

    async def set_user_name(self, tg_id: int, name: str) -> None:
        """Зберігає ім'я звіра й оновлює його як автора вже надісланих порад."""
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "INSERT INTO users (tg_id, real_name) VALUES ($1, $2) "
                "ON CONFLICT (tg_id) DO UPDATE SET real_name = EXCLUDED.real_name, "
                "updated_at = now()",
                tg_id,
                name,
            )
            await conn.execute(
                "UPDATE documents SET owner_name = $2 WHERE owner_tg_id = $1", tg_id, name
            )

    # --- документи -----------------------------------------------------------

    async def add_document(self, doc: NewDocument) -> int:
        doc_id: int = await self._pool.fetchval(
            "INSERT INTO documents (title, file_name, source, owner_tg_id, owner_name, level, "
            "place_id, summary, content, content_hash, file_hash) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11) RETURNING id",
            doc.title,
            doc.file_name,
            doc.source,
            doc.owner_tg_id,
            doc.owner_name,
            doc.scope.level.value,
            doc.scope.place_id,
            doc.summary,
            doc.content,
            doc.content_hash,
            doc.file_hash,
        )
        return doc_id

    async def get_document(self, doc_id: int) -> Document | None:
        docs = await self._select("WHERE id = $1", doc_id)
        return docs[0] if docs else None

    async def find_active_by_hash(self, content_hash: str) -> Document | None:
        docs = await self._select("WHERE content_hash = $1 AND is_active LIMIT 1", content_hash)
        return docs[0] if docs else None

    async def archive_file_hashes(self) -> set[str]:
        rows = await self._pool.fetch("SELECT file_hash FROM documents WHERE source = 'archive'")
        return {row["file_hash"] for row in rows}

    async def active_archive_documents(self) -> list[Document]:
        return await self._select("WHERE source = 'archive' AND is_active ORDER BY id")

    async def active_documents(self, scopes: Iterable[Scope] | None = None) -> list[Document]:
        if scopes is None:
            return await self._select("WHERE is_active ORDER BY id")
        conditions: list[str] = []
        params: list[Any] = []
        for scope in scopes:
            params.extend([scope.level.value, scope.place_id])
            conditions.append(
                f"(level = ${len(params) - 1} AND place_id IS NOT DISTINCT FROM ${len(params)})"
            )
        if not conditions:
            return []
        return await self._select(
            f"WHERE is_active AND ({' OR '.join(conditions)}) ORDER BY id", *params
        )

    async def user_documents(self, tg_id: int) -> list[Document]:
        return await self._select(
            "WHERE owner_tg_id = $1 AND is_active ORDER BY created_at DESC", tg_id
        )

    async def deactivate(self, doc_id: int, *, replaced_by: int | None = None) -> None:
        await self._pool.execute(
            "UPDATE documents SET is_active = FALSE, replaced_by = $2 WHERE id = $1",
            doc_id,
            replaced_by,
        )

    async def delete_document(self, doc_id: int) -> Document | None:
        """Вимикає документ; якщо він замінював попередню версію — повертає її в обіг."""
        async with self._pool.acquire() as conn, conn.transaction():
            restored = await conn.fetch(
                "UPDATE documents SET is_active = TRUE, replaced_by = NULL "
                "WHERE replaced_by = $1 AND NOT is_active RETURNING *",
                doc_id,
            )
            await conn.execute("UPDATE documents SET is_active = FALSE WHERE id = $1", doc_id)
        return _row_to_document(restored[0]) if restored else None

    async def update_scope(self, doc_id: int, scope: Scope) -> None:
        await self._pool.execute(
            "UPDATE documents SET level = $2, place_id = $3 WHERE id = $1",
            doc_id,
            scope.level.value,
            scope.place_id,
        )

    async def _select(self, where: str, *params: Any) -> list[Document]:
        rows = await self._pool.fetch(f"SELECT * FROM documents {where}", *params)
        return [_row_to_document(row) for row in rows]
