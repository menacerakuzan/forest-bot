from forest_bot.db import Database
from forest_bot.places import FOREST, Level, Scope
from forest_bot.qa import (
    SELECTION_THRESHOLD_CHARS,
    ConversationMemory,
    QAService,
    _with_same_topic,
)
from tests.conftest import FakeLLM, add_doc

DUB = Scope(Level.GLOVE, "dub")
DECIDUOUS = Scope(Level.PART, "deciduous")
RIVER = Scope(Level.PART, "river")


def service(db: Database, llm: FakeLLM) -> QAService:
    return QAService(
        db,
        llm,  # type: ignore[arg-type]
        answer_model="answer",
        fast_model="fast",
        timezone="Europe/Kyiv",
    )


async def test_answer_cites_archive_and_author(db: Database, llm: FakeLLM) -> None:
    archive_id = await add_doc(db, "Членські внески Під дубом", "4 жолуді", DUB)
    user_id = await add_doc(
        db, "Нові внески", "6 жолудів", DUB, source="telegram", owner=(1, "Їжак")
    )
    llm.queue("Answer", found=True, answer="6 жолудів на місяць.", source_ids=[user_id, archive_id])

    reply = await service(db, llm).ask("Скільки внесок під дубом?")

    assert reply.text.startswith("6 жолудів на місяць.")
    assert "«Членські внески Під дубом» — рукавичка «Під дубом», Лісовий архів" in reply.text
    assert "надіслав(-ла) Їжак" in reply.text
    assert not reply.needs_place
    # Модель знає автора кожного документа — щоб пояснити, хто що радить.
    prompt = llm.calls_for("Answer")[0]["user"]
    assert 'author="Їжак"' in prompt and 'author="Лісовий архів"' in prompt


async def test_only_applicable_levels_reach_the_model(db: Database, llm: FakeLLM) -> None:
    await add_doc(db, "Внески під дубом", "дуб", DUB)
    await add_doc(db, "Листяний ліс", "листя", DECIDUOUS)
    await add_doc(db, "Річка", "вода", RIVER)
    await add_doc(db, "Пожежа", "вогонь", FOREST)
    await add_doc(db, "Внески у соснах", "сосни", Scope(Level.GLOVE, "sosny"))
    llm.queue("Answer", found=True, answer="Так.", source_ids=[])

    await service(db, llm).ask("Що під дубом?")

    prompt = llm.calls_for("Answer")[0]["user"]
    assert all(word in prompt for word in ("дуб", "листя", "вогонь"))
    assert "вода" not in prompt and "сосни" not in prompt
    # Конкретніша порада йде першою.
    assert prompt.index("дуб") < prompt.index("листя") < prompt.index("вогонь")


async def test_verba_sees_both_parts(db: Database, llm: FakeLLM) -> None:
    await add_doc(db, "Утеплення листяного лісу", "листям", DECIDUOUS)
    await add_doc(db, "Утеплення біля річки", "очеретом", RIVER)
    llm.queue("Answer", found=True, answer="Поради розходяться.", source_ids=[])

    await service(db, llm).ask("Як утеплити рукавичку під вербою?")

    call = llm.calls_for("Answer")[0]
    assert "листям" in call["user"] and "очеретом" in call["user"]
    assert "розходяться" in call["system"]


async def test_no_place_and_nothing_general_asks_to_clarify(db: Database, llm: FakeLLM) -> None:
    await add_doc(db, "Пожежа", "вогонь", FOREST)
    llm.queue("Answer", found=False, answer="", source_ids=[])

    reply = await service(db, llm).ask("Скільки внесок?")

    assert reply.needs_place
    assert reply.question == "Скільки внесок?"


async def test_not_found_for_named_place(db: Database, llm: FakeLLM) -> None:
    await add_doc(db, "Пожежа", "вогонь", FOREST)
    llm.queue("Answer", found=False, answer="", source_ids=[])

    reply = await service(db, llm).ask("Хто президент під дубом?")

    assert "не знайшла" in reply.text and "«Під дубом»" in reply.text
    assert not reply.needs_place


async def test_large_corpus_goes_through_selection(db: Database, llm: FakeLLM) -> None:
    filler = "текст " * (SELECTION_THRESHOLD_CHARS // 6)
    fees = await add_doc(db, "Членські внески Під дубом", "4 жолуді", DUB)
    await add_doc(db, "Статут Під дубом", filler, DUB)
    await add_doc(db, "Утеплення листяного лісу", filler, DECIDUOUS)
    llm.queue("Selection", doc_ids=[fees, 999])
    llm.queue("Answer", found=True, answer="4 жолуді.", source_ids=[fees])

    reply = await service(db, llm).ask("Скільки внесок під дубом?")

    assert "4 жолуді." in reply.text
    assert "Статут" not in llm.calls_for("Answer")[0]["user"]


async def test_history_is_passed_for_follow_ups(db: Database, llm: FakeLLM) -> None:
    await add_doc(db, "Внески під вербою", "2 жмені", Scope(Level.GLOVE, "verba"))
    llm.queue("Answer", found=True, answer="2 жмені.", source_ids=[])
    memory = ConversationMemory()
    memory.add(1, "Скільки внесок під дубом?")

    await service(db, llm).ask("а під вербою?", memory.render(1))

    assert "Скільки внесок під дубом?" in llm.calls_for("Answer")[0]["user"]


async def test_answer_text_is_html_escaped(db: Database, llm: FakeLLM) -> None:
    await add_doc(db, "Пожежа", "вогонь", FOREST)
    llm.queue("Answer", found=True, answer="<script> **важливо**", source_ids=[])

    reply = await service(db, llm).ask("Що робити при пожежі?")

    assert reply.text == "&lt;script&gt; <b>важливо</b>"


async def test_same_topic_documents_from_other_levels_are_added(db: Database) -> None:
    river = await add_doc(db, "Як утеплити рукавичку біля річки", "очерет", RIVER)
    await add_doc(db, "Як утеплити рукавичку в листяному лісі", "листя", DECIDUOUS)
    await add_doc(db, "Небезпеки біля річки", "щука", RIVER)
    available = await db.active_documents()
    selected = [d for d in available if d.id == river]

    titles = [d.title for d in _with_same_topic(selected, available)]

    assert titles == [
        "Як утеплити рукавичку біля річки",
        "Як утеплити рукавичку в листяному лісі",
    ]


def test_memory_keeps_only_recent_questions() -> None:
    memory = ConversationMemory(max_questions=2)
    for question in ("перше", "друге", "третє"):
        memory.add(7, question)
    assert memory.render(7) == "- друге\n- третє"
    assert memory.render(8) == ""
