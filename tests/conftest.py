from collections import defaultdict, deque
from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import BaseModel

from forest_bot.db import Database, NewDocument
from forest_bot.ingest import content_hash
from forest_bot.places import Scope


class FakeLLM:
    """Підмінює Groq: повертає заздалегідь задані відповіді для кожної схеми."""

    def __init__(self) -> None:
        self.responses: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
        self.calls: list[dict[str, Any]] = []

    def queue(self, schema: str, **response: Any) -> None:
        self.responses[schema].append(response)

    async def structured(self, *, schema: type[BaseModel], **kwargs: Any) -> BaseModel:
        self.calls.append({"schema": schema.__name__, **kwargs})
        queued = self.responses[schema.__name__]
        if not queued:
            raise AssertionError(f"unexpected LLM call for {schema.__name__}")
        return schema.model_validate(queued.popleft())

    def calls_for(self, schema: str) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["schema"] == schema]


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
async def db() -> AsyncIterator[Database]:
    database = await Database.open(":memory:")
    yield database
    await database.close()


async def add_doc(
    db: Database,
    title: str,
    content: str,
    scope: Scope,
    *,
    source: str = "archive",
    owner: tuple[int, str] | None = None,
) -> int:
    return await db.add_document(
        NewDocument(
            title=title,
            file_name=f"{title}.txt",
            source=source,
            scope=scope,
            summary=content[:60],
            content=content,
            content_hash=content_hash(content),
            owner_tg_id=owner[0] if owner else None,
            owner_name=owner[1] if owner else None,
        )
    )
