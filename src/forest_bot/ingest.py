"""Приймання документів: класифікація за місцем, заміна старих версій, стартовий архів."""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from forest_bot import prompts
from forest_bot.db import Database, Document, NewDocument
from forest_bot.extract import (
    SUPPORTED_SUFFIXES,
    EmptyDocumentError,
    UnsupportedFormatError,
    extract_text,
)
from forest_bot.llm import LLM
from forest_bot.places import PLACES, Level, PlaceId, Scope

log = logging.getLogger(__name__)

# Для класифікації досить початку документа: звірі пишуть, для кого порада, на початку.
CLASSIFY_EXCERPT_CHARS = 4000


class Classification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    level: Literal["forest", "part", "glove", "unknown"]
    place: PlaceId | None
    mixed_places: bool
    summary: str

    def scope(self) -> Scope | None:
        """Рівень дії поради або None, якщо з документа це незрозуміло."""
        if self.level == "forest":
            return Scope(Level.FOREST)
        if self.level == "unknown" or self.place is None:
            return None
        level = Level(self.level)
        if PLACES[self.place].level is not level:
            return None
        return Scope(level, self.place)


class Replacement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    replaces_id: int | None


class Draft(BaseModel):
    """Документ, що пройшов класифікацію, але ще не збережений (зберігається у стані FSM)."""

    title: str
    file_name: str
    summary: str
    content: str
    content_hash: str
    owner_tg_id: int
    owner_name: str


@dataclass(frozen=True)
class Duplicate:
    existing: Document


@dataclass(frozen=True)
class MixedPlaces:
    """Документ схожий на поради для кількох місць — автора треба перепитати."""

    draft: Draft


@dataclass(frozen=True)
class NeedsScope:
    draft: Draft


@dataclass(frozen=True)
class Ready:
    draft: Draft
    scope: Scope


PrepareResult = Duplicate | MixedPlaces | NeedsScope | Ready


@dataclass(frozen=True)
class Stored:
    document: Document
    replaced: Document | None


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class Ingestor:
    def __init__(self, db: Database, llm: LLM, *, model: str) -> None:
        self._db = db
        self._llm = llm
        self._model = model

    async def classify(self, text: str, file_name: str) -> Classification:
        return await self._llm.structured(
            model=self._model,
            system=prompts.CLASSIFY,
            user=f"Файл: {file_name}\n\n<document>\n{text[:CLASSIFY_EXCERPT_CHARS]}\n</document>",
            schema=Classification,
        )

    async def prepare(
        self, *, text: str, file_name: str, owner_tg_id: int, owner_name: str
    ) -> PrepareResult:
        digest = content_hash(text)
        if existing := await self._db.find_active_by_hash(digest):
            return Duplicate(existing)
        classification = await self.classify(text, file_name)
        draft = Draft(
            title=classification.title.strip() or Path(file_name).stem,
            file_name=file_name,
            summary=classification.summary.strip(),
            content=text,
            content_hash=digest,
            owner_tg_id=owner_tg_id,
            owner_name=owner_name,
        )
        if classification.mixed_places:
            return MixedPlaces(draft)
        scope = classification.scope()
        return Ready(draft, scope) if scope else NeedsScope(draft)

    async def store(self, draft: Draft, scope: Scope) -> Stored:
        replaced = await self._find_replaced(draft, scope)
        doc_id = await self._db.add_document(
            NewDocument(
                title=draft.title,
                file_name=draft.file_name,
                source="telegram",
                scope=scope,
                summary=draft.summary,
                content=draft.content,
                content_hash=draft.content_hash,
                owner_tg_id=draft.owner_tg_id,
                owner_name=draft.owner_name,
            )
        )
        if replaced:
            await self._db.deactivate(replaced.id, replaced_by=doc_id)
            log.info("document %s replaced %s", doc_id, replaced.id)
        document = await self._db.get_document(doc_id)
        assert document is not None
        return Stored(document, replaced)

    async def _find_replaced(self, draft: Draft, scope: Scope) -> Document | None:
        candidates = await self._db.active_documents([scope])
        if not candidates:
            return None
        catalog = "\n".join(f"[{d.id}] «{d.title}» — {d.summary}" for d in candidates)
        result = await self._llm.structured(
            model=self._model,
            system=prompts.REPLACES.format(scope=scope.label),
            user=(
                f"Наявні документи:\n{catalog}\n\n"
                f"Новий документ «{draft.title}» — {draft.summary}\n"
                f"<document>\n{draft.content[:CLASSIFY_EXCERPT_CHARS]}\n</document>"
            ),
            schema=Replacement,
        )
        return next((d for d in candidates if d.id == result.replaces_id), None)

    async def load_archive(self, archive_dir: Path, index_path: Path) -> None:
        """Завантажує «Лісовий архів»: нові файли додає, змінені замінюють стару версію.

        Метадані (рівень, місце, опис) беруться з індексу за хешем файлу; файли, яких немає
        в індексі, класифікуються через LLM. Відомі файли на наступних стартах пропускаються.
        """
        index = _read_index(index_path)
        known = await self._db.archive_file_hashes()
        active = {d.file_name: d for d in await self._db.active_archive_documents()}
        files = sorted(
            p
            for p in archive_dir.iterdir()  # noqa: ASYNC240 — разово під час старту
            if p.suffix.lower() in SUPPORTED_SUFFIXES
        )
        added = 0
        for path in files:
            data = path.read_bytes()
            file_hash = hashlib.sha256(data).hexdigest()
            if file_hash in known:
                continue
            try:
                # OCR сканів — важка синхронна робота, не блокуємо цикл подій.
                text = await asyncio.to_thread(extract_text, data, path.suffix.lower())
            except (UnsupportedFormatError, EmptyDocumentError):
                log.warning("skipping unreadable archive file %s", path.name)
                continue
            meta = index.get(file_hash)
            if meta is None:
                log.info("classifying archive file %s", path.name)
                meta = (await self.classify(text, path.name)).model_dump()
                index[file_hash] = meta | {"file_name": path.name}
                _write_index(index_path, index)
            classification = Classification.model_validate(
                {k: v for k, v in meta.items() if k != "file_name"}
            )
            scope = classification.scope()
            if scope is None:
                log.warning("archive file %s has unclear scope, treating as whole forest", path)
                scope = Scope(Level.FOREST)
            doc_id = await self._db.add_document(
                NewDocument(
                    title=path.stem,
                    file_name=path.name,
                    source="archive",
                    scope=scope,
                    summary=classification.summary,
                    content=text,
                    content_hash=content_hash(text),
                    file_hash=file_hash,
                )
            )
            if previous := active.pop(path.name, None):
                await self._db.deactivate(previous.id, replaced_by=doc_id)
                log.info("archive file %s updated", path.name)
            added += 1
        log.info("archive ready: %d files, %d added/updated", len(files), added)


def _read_index(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    data: dict[str, dict[str, object]] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _write_index(path: Path, index: dict[str, dict[str, object]]) -> None:
    try:
        path.write_text(
            json.dumps(
                dict(sorted(index.items(), key=lambda kv: str(kv[1].get("file_name")))),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError:
        log.debug("archive index is read-only, not updating %s", path)
