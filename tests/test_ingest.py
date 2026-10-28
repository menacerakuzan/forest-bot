import hashlib
import json
from pathlib import Path

from forest_bot.db import Database
from forest_bot.ingest import Duplicate, Ingestor, MixedPlaces, NeedsScope, Ready
from forest_bot.places import FOREST, Level, Scope
from tests.conftest import FakeLLM, add_doc

DUB = Scope(Level.GLOVE, "dub")


def classification(level: str, place: str | None, **extra: object) -> dict[str, object]:
    return {
        "title": "Внески",
        "level": level,
        "place": place,
        "mixed_places": False,
        "summary": "Про внески",
    } | extra


async def prepare(ingestor: Ingestor, text: str) -> object:
    return await ingestor.prepare(text=text, file_name="d.txt", owner_tg_id=1, owner_name="Їжак")


async def test_ready_when_scope_is_clear(db: Database, llm: FakeLLM) -> None:
    llm.queue("Classification", **classification("glove", "dub"))
    result = await prepare(Ingestor(db, llm, model="m"), "Внески під дубом")  # type: ignore[arg-type]
    assert isinstance(result, Ready)
    assert result.scope == DUB
    assert result.draft.owner_name == "Їжак"


async def test_unclear_scope_asks_sender(db: Database, llm: FakeLLM) -> None:
    llm.queue("Classification", **classification("unknown", None))
    result = await prepare(Ingestor(db, llm, model="m"), "Порада без місця")  # type: ignore[arg-type]
    assert isinstance(result, NeedsScope)


async def test_inconsistent_classification_is_treated_as_unclear(
    db: Database, llm: FakeLLM
) -> None:
    llm.queue("Classification", **classification("part", "dub"))
    result = await prepare(Ingestor(db, llm, model="m"), "Щось")  # type: ignore[arg-type]
    assert isinstance(result, NeedsScope)


async def test_mixed_places_ask_the_author(db: Database, llm: FakeLLM) -> None:
    llm.queue("Classification", **classification("glove", "dub", mixed_places=True))
    result = await prepare(Ingestor(db, llm, model="m"), "Під дубом… У соснах…")  # type: ignore[arg-type]
    assert isinstance(result, MixedPlaces)
    assert result.draft.content == "Під дубом… У соснах…"


async def test_duplicate_skips_llm(db: Database, llm: FakeLLM) -> None:
    await add_doc(db, "Внески", "Внесок 4 жолуді", DUB)
    result = await prepare(Ingestor(db, llm, model="m"), "Внесок 4 жолуді")  # type: ignore[arg-type]
    assert isinstance(result, Duplicate)
    assert llm.calls == []


async def test_new_version_replaces_old(db: Database, llm: FakeLLM) -> None:
    old_id = await add_doc(db, "Внески", "Внесок 4 жолуді", DUB)
    ingestor = Ingestor(db, llm, model="m")  # type: ignore[arg-type]
    llm.queue("Classification", **classification("glove", "dub"))
    llm.queue("Replacement", replaces_id=old_id)
    result = await prepare(ingestor, "Внесок тепер 6 жолудів")
    assert isinstance(result, Ready)

    stored = await ingestor.store(result.draft, result.scope)

    assert stored.replaced is not None and stored.replaced.id == old_id
    active = await db.active_documents([DUB])
    assert [d.content for d in active] == ["Внесок тепер 6 жолудів"]
    assert active[0].owner_name == "Їжак"


async def test_no_replacement_check_without_candidates(db: Database, llm: FakeLLM) -> None:
    ingestor = Ingestor(db, llm, model="m")  # type: ignore[arg-type]
    llm.queue("Classification", **classification("forest", None))
    result = await prepare(ingestor, "Для всього лісу")
    assert isinstance(result, Ready)
    stored = await ingestor.store(result.draft, FOREST)
    assert stored.replaced is None
    assert llm.calls_for("Replacement") == []


async def test_archive_uses_index_and_loads_once(
    db: Database, llm: FakeLLM, tmp_path: Path
) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    indexed = archive / "Внески Під дубом.txt"
    indexed.write_text("Внесок — 4 жолуді", encoding="utf-8")
    fresh = archive / "Пожежа.txt"
    fresh.write_text("Для всіх мешканців лісу: біжіть до річки", encoding="utf-8")
    index_path = tmp_path / "index.json"
    file_hash = hashlib.sha256(indexed.read_bytes()).hexdigest()
    index_path.write_text(
        json.dumps({file_hash: classification("glove", "dub", file_name=indexed.name)}),
        encoding="utf-8",
    )
    ingestor = Ingestor(db, llm, model="m")  # type: ignore[arg-type]
    llm.queue("Classification", **classification("forest", None))

    await ingestor.load_archive(archive, index_path)
    await ingestor.load_archive(archive, index_path)  # повторний старт нічого не дублює

    documents = await db.active_documents()
    assert {(d.title, d.scope) for d in documents} == {
        ("Внески Під дубом", DUB),
        ("Пожежа", FOREST),
    }
    assert all(d.from_archive for d in documents)
    assert len(llm.calls_for("Classification")) == 1
    assert len(json.loads(index_path.read_text(encoding="utf-8"))) == 2


async def test_deleting_new_version_restores_previous(db: Database, llm: FakeLLM) -> None:
    old_id = await add_doc(db, "Внески", "Внесок 4 жолуді", DUB)
    ingestor = Ingestor(db, llm, model="m")  # type: ignore[arg-type]
    llm.queue("Classification", **classification("glove", "dub"))
    llm.queue("Replacement", replaces_id=old_id)
    result = await prepare(ingestor, "Внесок тепер 6 жолудів")
    assert isinstance(result, Ready)
    stored = await ingestor.store(result.draft, result.scope)

    restored = await db.delete_document(stored.document.id)

    assert restored is not None and restored.id == old_id
    assert [d.id for d in await db.active_documents([DUB])] == [old_id]


async def test_changed_archive_file_replaces_previous_version(
    db: Database, llm: FakeLLM, tmp_path: Path
) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "Внески.txt").write_text("Внесок — 4 жолуді", encoding="utf-8")
    index = tmp_path / "index.json"
    ingestor = Ingestor(db, llm, model="m")  # type: ignore[arg-type]
    for _ in range(2):
        llm.queue("Classification", **classification("glove", "dub"))

    await ingestor.load_archive(archive, index)
    (archive / "Внески.txt").write_text("Внесок — 6 жолудів", encoding="utf-8")
    await ingestor.load_archive(archive, index)

    assert [d.content for d in await db.active_documents()] == ["Внесок — 6 жолудів"]


async def test_renaming_updates_author_of_existing_documents(db: Database) -> None:
    await db.set_user_name(5, "Ілля")
    doc_id = await add_doc(db, "Порада", "Текст", DUB, source="telegram", owner=(5, "Ілля"))

    await db.set_user_name(5, "Ілля Довженко")

    assert await db.get_user_name(5) == "Ілля Довженко"
    document = await db.get_document(doc_id)
    assert document is not None and document.owner_name == "Ілля Довженко"
