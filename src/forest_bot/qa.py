"""Відповіді на питання: місце → доречні документи → відповідь із джерелами."""

import html
import logging
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict

from forest_bot import prompts
from forest_bot.db import Database, Document
from forest_bot.extract import split_into_chunks
from forest_bot.llm import LLM
from forest_bot.places import PLACES, applicable_scopes, detect_places

log = logging.getLogger(__name__)

# Якщо доречні документи вкладаються в цей обсяг — віддаємо їх цілими, без відбору.
SELECTION_THRESHOLD_CHARS = 8000
# Скільки тексту документів максимум іде в один запит відповіді (ліміти безкоштовного Groq).
CONTEXT_BUDGET_CHARS = 12000
LARGE_DOCUMENT_CHARS = 4000
MAX_PLACES_PER_QUESTION = 3

MONTHS = [
    "січня",
    "лютого",
    "березня",
    "квітня",
    "травня",
    "червня",
    "липня",
    "серпня",
    "вересня",
    "жовтня",
    "листопада",
    "грудня",
]


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_ids: list[int]


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    found: bool
    answer: str
    source_ids: list[int]


@dataclass(frozen=True)
class Reply:
    """Готова відповідь звірові (HTML для Telegram)."""

    text: str
    needs_place: bool = False
    question: str = ""


class ConversationMemory:
    """Кілька останніх питань звіра — щоб розуміти уточнення на кшталт «а під вербою?».

    Відповіді бота сюди свідомо не потрапляють: джерелом фактів мають бути лише документи.
    """

    def __init__(self, max_questions: int = 3) -> None:
        self._questions: dict[int, deque[str]] = defaultdict(lambda: deque(maxlen=max_questions))

    def add(self, user_id: int, question: str) -> None:
        self._questions[user_id].append(question[:300])

    def render(self, user_id: int) -> str:
        return "\n".join(f"- {q}" for q in self._questions[user_id])


class QAService:
    def __init__(
        self, db: Database, llm: LLM, *, answer_model: str, fast_model: str, timezone: str
    ) -> None:
        self._db = db
        self._llm = llm
        self._answer_model = answer_model
        self._fast_model = fast_model
        self._tz = ZoneInfo(timezone)

    def local_date(self, moment: datetime) -> str:
        """Дата в часовому поясі лісу (у базі час зберігається в UTC)."""
        return _format_date(moment, self._tz)

    async def ask(self, question: str, history: str = "") -> Reply:
        places = detect_places(question)[:MAX_PLACES_PER_QUESTION]
        if not places:
            # Місце не назване: шукаємо серед порад для всього лісу, інакше перепитуємо.
            reply = await self.answer(question, [], history)
            return reply or _clarify(question)
        reply = await self.answer(question, places, history)
        return reply or Reply(_not_found_text(places))

    async def ask_for_place(self, question: str, place_id: str, history: str = "") -> Reply:
        """Питання, для якого звір обрав місце кнопкою після уточнення."""
        reply = await self.answer(question, [place_id], history)
        return reply or Reply(_not_found_text([place_id]))

    async def answer(self, question: str, places: list[str], history: str = "") -> Reply | None:
        """Відповідь для названих місць (порожній список — весь ліс); None — нічого не знайшлося."""
        targets: list[str | None] = [*places] or [None]
        scopes = list(dict.fromkeys(s for place in targets for s in applicable_scopes(place)))
        available = await self._db.active_documents(scopes)
        if not available:
            return None
        documents = available
        if sum(len(d.content) for d in available) > SELECTION_THRESHOLD_CHARS:
            selected = await self._select(question, available, history)
            documents = _with_same_topic(selected, available)
            log.info("selected for %r @ %s: %s", question, places, [d.title for d in documents])
            if not documents:
                return None
        answer = await self._answer(question, places, documents, history)
        if not answer.found or not answer.answer.strip():
            return None
        by_id = {d.id: d for d in documents}
        sources = [by_id[i] for i in dict.fromkeys(answer.source_ids) if i in by_id]
        return Reply(_format_answer(answer.answer, sources, self._tz))

    async def _select(
        self, question: str, documents: list[Document], history: str
    ) -> list[Document]:
        catalog = "\n".join(
            f"[{d.id}] ({d.scope.label}) «{d.title}» — {d.summary}" for d in documents
        )
        selection = await self._llm.structured(
            model=self._fast_model,
            system=prompts.SELECT,
            user=f"Каталог документів:\n{catalog}\n\n{_question_block(question, history)}",
            schema=Selection,
        )
        by_id = {d.id: d for d in documents}
        return [by_id[i] for i in dict.fromkeys(selection.doc_ids) if i in by_id]

    async def _answer(
        self, question: str, places: list[str], documents: list[Document], history: str
    ) -> Answer:
        today = datetime.now(self._tz)
        system = prompts.ANSWER.format(
            today=f"{today.day} {MONTHS[today.month - 1]} {today.year}",
            place_note=_place_note(places),
        )
        return await self._llm.structured(
            model=self._answer_model,
            system=system,
            user=f"{_build_context(question, documents)}\n\n{_question_block(question, history)}",
            schema=Answer,
            reasoning_effort="medium",
        )


def _place_note(places: list[str]) -> str:
    if not places:
        return "Звір питає про ліс загалом — це поради для всіх мешканців лісу."
    if len(places) == 1:
        note = f"Звір питає про: {PLACES[places[0]].label}. Відповідай саме для цього місця."
    else:
        lines = [
            f"- {PLACES[p].name}: " + ", ".join(s.label for s in applicable_scopes(p))
            for p in places
        ]
        note = (
            "Звір питає про кілька місць. Відповідай для кожного окремо (почни з назви місця). "
            "Для кожного місця діють лише документи таких рівнів:\n" + "\n".join(lines)
        )
    if "verba" in places:
        note += " " + prompts.VERBA_NOTE
    return note


def _build_context(question: str, documents: list[Document]) -> str:
    """Документи для промпту — від найконкретнішого рівня, в межах бюджету символів."""
    blocks, used = [], 0
    for doc in sorted(documents, key=lambda d: d.scope.priority):
        text = doc.content
        if len(text) > LARGE_DOCUMENT_CHARS:
            text = _relevant_excerpt(text, question, LARGE_DOCUMENT_CHARS)
        if used + len(text) > CONTEXT_BUDGET_CHARS:
            continue
        used += len(text)
        author = "Лісовий архів" if doc.from_archive else doc.owner_name
        blocks.append(
            f'<document id="{doc.id}" level="{doc.scope.label}" '
            f'priority="{doc.scope.priority + 1}" title="{doc.title}" author="{author}">'
            f"\n{text}\n</document>"
        )
    return "\n\n".join(blocks)


def _stems(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"\w+", text.lower()) if len(w) > 2}


# Слова в назвах, що описують місце, а не тему документа.
_PLACE_WORDS = _stems(
    " ".join(p.name for p in PLACES.values())
    + " ліс лісу лісі лісі рукавичка рукавички рукавичці рукавичкою поруч біля для річки "
    "хвойного хвойному листяного листяному"
)


def _topic(doc: Document) -> set[str]:
    return _stems(doc.title) - _PLACE_WORDS


def _with_same_topic(selected: list[Document], available: list[Document]) -> list[Document]:
    """Додає документи тієї ж теми з інших рівнів — щоб модель бачила, що конкретне скасовує."""
    topics = [_topic(d) for d in selected]
    chosen = {d.id for d in selected}
    extra = [
        d
        for d in available
        if d.id not in chosen and any(_topic(d) & topic for topic in topics if topic)
    ]
    return selected + extra


def _relevant_excerpt(text: str, question: str, budget: int) -> str:
    """Для великих документів (путівників) беремо шматки, найближчі до питання."""
    chunks = split_into_chunks(text)
    question_stems = _stems(question)
    ranked = sorted(
        range(len(chunks)), key=lambda i: len(_stems(chunks[i]) & question_stems), reverse=True
    )
    chosen, used = [], 0
    for i in ranked:
        if used + len(chunks[i]) > budget:
            continue
        chosen.append(i)
        used += len(chunks[i])
    return "\n…\n".join(chunks[i] for i in sorted(chosen))


def _question_block(question: str, history: str) -> str:
    if not history:
        return f"Питання звіра: {question}"
    return (
        f"Попередні питання звіра (лише щоб зрозуміти уточнення, не джерело фактів):\n{history}\n\n"
        f"Питання звіра: {question}"
    )


def _format_date(moment: datetime, tz: ZoneInfo) -> str:
    return moment.astimezone(tz).strftime("%d.%m.%Y")


def source_line(doc: Document, tz: ZoneInfo) -> str:
    title = html.escape(doc.title)
    where = html.escape(doc.scope.label)
    if doc.from_archive:
        return f"• «{title}» — {where}, Лісовий архів"
    author = html.escape(doc.owner_name or "невідомий звір")
    return f"• «{title}» — {where}; надіслав(-ла) {author}, {_format_date(doc.created_at, tz)}"


def _format_answer(answer: str, sources: list[Document], tz: ZoneInfo) -> str:
    body = html.escape(answer.strip(), quote=False)
    body = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", body)
    if not sources:
        return body
    lines = "\n".join(source_line(d, tz) for d in sources)
    return f"{body}\n\n📄 <b>Джерела:</b>\n{lines}"


def _not_found_text(places: list[str]) -> str:
    names = ", ".join(f"«{PLACES[p].name}»" for p in places)
    where = f" для {names}" if places else ""
    return (
        f"Я не знайшла про це порад{html.escape(where)} 🍂\n"
        "Якщо ти знаєш відповідь — надішли документ з порадою, і я її запам'ятаю."
    )


def _clarify(question: str) -> Reply:
    return Reply(
        "Серед порад для всього лісу я цього не знайшла 🍂\n"
        "Якщо питаєш про конкретне місце — обери рукавичку чи частину лісу або напиши "
        "питання з її назвою 👇",
        needs_place=True,
        question=question,
    )
