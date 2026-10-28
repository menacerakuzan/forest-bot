"""Сховище звірів і документів у SQLite."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import aiosqlite

from forest_bot.places import Level, Scope

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    tg_id      INTEGER PRIMARY KEY,
    real_name  TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    title        TEXT NOT NULL,
    file_name    TEXT NOT NULL,
    source       TEXT NOT NULL CHECK (source IN ('archive', 'telegram')),
    owner_tg_id  INTEGER,
    owner_name   TEXT,
    level        TEXT NOT NULL CHECK (level IN ('forest', 'part', 'glove')),
    place_id     TEXT,
    summary      TEXT NOT NULL,
    content      TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    file_hash    TEXT,
    created_at   TEXT NOT NULL,
    is_active    INTEGER NOT NULL DEFAULT 1,
    replaced_by  INTEGER REFERENCES documents (id)
);

CREATE INDEX IF NOT EXISTS ix_documents_scope ON documents (is_active, level, place_id);
CREATE INDEX IF NOT EXISTS ix_documents_owner ON documents (owner_tg_id, is_active);
CREATE INDEX IF NOT EXISTS ix_documents_hash ON documents (content_hash);
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


def _row_to_document(row: aiosqlite.Row) -> Document:
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
        created_at=datetime.fromisoformat(row["created_at"]),
        is_active=bool(row["is_active"]),
    )


class Database:
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    @classmethod
    async def open(cls, path: Path | str) -> Self:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(path)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA foreign_keys=ON")
        await conn.executescript(SCHEMA)
        await conn.commit()
        return cls(conn)

    async def close(self) -> None:
        await self._conn.close()

    # --- звірі ---------------------------------------------------------------

    async def get_user_name(self, tg_id: int) -> str | None:
        async with self._conn.execute(
            "SELECT real_name FROM users WHERE tg_id = ?", (tg_id,)
        ) as cur:
            row = await cur.fetchone()
        return row["real_name"] if row else None

    async def set_user_name(self, tg_id: int, name: str) -> None:
        await self._conn.execute(
            "INSERT INTO users (tg_id, real_name, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT (tg_id) DO UPDATE SET real_name = excluded.real_name, "
            "updated_at = excluded.updated_at",
            (tg_id, name, datetime.now(UTC).isoformat()),
        )
        await self._conn.commit()

    # --- документи -----------------------------------------------------------

    async def add_document(self, doc: NewDocument, *, created_at: datetime | None = None) -> int:
        cur = await self._conn.execute(
            "INSERT INTO documents (title, file_name, source, owner_tg_id, owner_name, level, "
            "place_id, summary, content, content_hash, file_hash, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
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
                (created_at or datetime.now(UTC)).isoformat(),
            ),
        )
        await self._conn.commit()
        assert cur.lastrowid is not None
        return cur.lastrowid

    async def get_document(self, doc_id: int) -> Document | None:
        docs = await self._select("WHERE id = ?", (doc_id,))
        return docs[0] if docs else None

    async def find_active_by_hash(self, content_hash: str) -> Document | None:
        docs = await self._select("WHERE content_hash = ? AND is_active = 1", (content_hash,))
        return docs[0] if docs else None

    async def archive_file_hashes(self) -> set[str]:
        async with self._conn.execute(
            "SELECT file_hash FROM documents WHERE source = 'archive'"
        ) as cur:
            return {row["file_hash"] for row in await cur.fetchall()}

    async def active_archive_documents(self) -> list[Document]:
        return await self._select("WHERE source = 'archive' AND is_active = 1 ORDER BY id")

    async def active_documents(self, scopes: Iterable[Scope] | None = None) -> list[Document]:
        if scopes is None:
            return await self._select("WHERE is_active = 1 ORDER BY id")
        conditions, params = [], []
        for scope in scopes:
            conditions.append("(level = ? AND place_id IS ?)")
            params.extend([scope.level.value, scope.place_id])
        if not conditions:
            return []
        return await self._select(
            f"WHERE is_active = 1 AND ({' OR '.join(conditions)}) ORDER BY id", params
        )

    async def user_documents(self, tg_id: int) -> list[Document]:
        return await self._select(
            "WHERE owner_tg_id = ? AND is_active = 1 ORDER BY created_at DESC", (tg_id,)
        )

    async def deactivate(self, doc_id: int, *, replaced_by: int | None = None) -> None:
        await self._conn.execute(
            "UPDATE documents SET is_active = 0, replaced_by = ? WHERE id = ?",
            (replaced_by, doc_id),
        )
        await self._conn.commit()

    async def delete_document(self, doc_id: int) -> Document | None:
        """Вимикає документ; якщо він замінював попередню версію — повертає її в обіг."""
        restored = await self._select("WHERE replaced_by = ? AND is_active = 0", (doc_id,))
        await self._conn.execute(
            "UPDATE documents SET is_active = 1, replaced_by = NULL WHERE replaced_by = ?",
            (doc_id,),
        )
        await self._conn.execute("UPDATE documents SET is_active = 0 WHERE id = ?", (doc_id,))
        await self._conn.commit()
        return restored[0] if restored else None

    async def update_scope(self, doc_id: int, scope: Scope) -> None:
        await self._conn.execute(
            "UPDATE documents SET level = ?, place_id = ? WHERE id = ?",
            (scope.level.value, scope.place_id, doc_id),
        )
        await self._conn.commit()

    async def _select(self, where: str, params: Iterable[Any] = ()) -> list[Document]:
        async with self._conn.execute(f"SELECT * FROM documents {where}", tuple(params)) as cur:
            return [_row_to_document(row) for row in await cur.fetchall()]
