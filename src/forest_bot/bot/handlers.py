"""Обробники повідомлень Telegram."""

import asyncio
import html
import logging

from aiogram import Bot, F, Router
from aiogram.enums import ChatType
from aiogram.filters import Command, CommandStart, ExceptionTypeFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, ErrorEvent, Message
from aiogram.utils.chat_action import ChatActionSender

from forest_bot.bot import keyboards as kb
from forest_bot.bot import texts
from forest_bot.db import Database
from forest_bot.extract import (
    SUPPORTED_SUFFIXES,
    EmptyDocumentError,
    extract_text,
    resolve_suffix,
)
from forest_bot.ingest import Draft, Duplicate, Ingestor, MixedPlaces, NeedsScope, Ready, Stored
from forest_bot.llm import LLMError
from forest_bot.places import Scope
from forest_bot.qa import ConversationMemory, QAService, Reply

log = logging.getLogger(__name__)

MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_NAME_LENGTH = 40

router = Router(name="forest")
router.message.filter(F.chat.type == ChatType.PRIVATE)


class Upload(StatesGroup):
    waiting_name = State()
    waiting_scope = State()


class Profile(StatesGroup):
    waiting_name = State()


# --- команди ------------------------------------------------------------------


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(texts.WELCOME)


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(texts.HELP)


@router.message(Command("name"))
async def cmd_name(message: Message, state: FSMContext) -> None:
    await state.set_state(Profile.waiting_name)
    await message.answer(texts.ASK_NEW_NAME)


@router.message(Command("my_docs"))
async def cmd_my_docs(message: Message, db: Database) -> None:
    assert message.from_user
    documents = await db.user_documents(message.from_user.id)
    if not documents:
        await message.answer(texts.MY_DOCS_EMPTY)
        return
    lines = [
        f"• «{html.escape(d.title)}» — {html.escape(d.scope.label)}, "
        f"{d.created_at.strftime('%d.%m.%Y')}"
        for d in documents
    ]
    await message.answer(
        texts.MY_DOCS_HEADER + "\n".join(lines), reply_markup=kb.my_docs_keyboard(documents)
    )


# --- ім'я автора ------------------------------------------------------------------


def _clean_name(text: str | None) -> str | None:
    name = " ".join((text or "").split())
    if not name or name.startswith("/") or len(name) > MAX_NAME_LENGTH:
        return None
    return name


@router.message(Profile.waiting_name, F.text)
async def on_new_name(message: Message, state: FSMContext, db: Database) -> None:
    assert message.from_user
    name = _clean_name(message.text)
    if name is None:
        await message.answer(texts.BAD_NAME)
        return
    await db.set_user_name(message.from_user.id, name)
    await state.clear()
    await message.answer(texts.NAME_SAVED.format(name=html.escape(name)))


@router.message(Upload.waiting_name, F.text)
async def on_author_name(
    message: Message, state: FSMContext, bot: Bot, db: Database, ingestor: Ingestor
) -> None:
    assert message.from_user
    name = _clean_name(message.text)
    if name is None:
        await message.answer(texts.BAD_NAME)
        return
    await db.set_user_name(message.from_user.id, name)
    data = await state.get_data()
    await state.clear()
    await message.answer(texts.NAME_SAVED.format(name=html.escape(name)))
    await _process_document(
        message, state, bot, ingestor, data["file_id"], data["file_name"], data["suffix"], name
    )


# --- документи ------------------------------------------------------------------


@router.message(F.document)
async def on_document(
    message: Message, state: FSMContext, bot: Bot, db: Database, ingestor: Ingestor
) -> None:
    assert message.document and message.from_user
    document = message.document
    suffix = resolve_suffix(document.file_name, document.mime_type)
    file_name = document.file_name or f"документ{suffix}"
    if suffix not in SUPPORTED_SUFFIXES:
        await message.answer(texts.UNSUPPORTED_FORMAT)
        return
    if (document.file_size or 0) > MAX_FILE_BYTES:
        await message.answer(texts.TOO_BIG)
        return

    name = await db.get_user_name(message.from_user.id)
    if name is None:
        await state.set_state(Upload.waiting_name)
        await state.update_data(file_id=document.file_id, file_name=file_name, suffix=suffix)
        await message.answer(texts.ASK_NAME)
        return
    await _process_document(
        message, state, bot, ingestor, document.file_id, file_name, suffix, name
    )


async def _process_document(
    message: Message,
    state: FSMContext,
    bot: Bot,
    ingestor: Ingestor,
    file_id: str,
    file_name: str,
    suffix: str,
    owner_name: str,
) -> None:
    assert message.from_user
    status = await message.answer(texts.READING)
    async with ChatActionSender.upload_document(bot=bot, chat_id=message.chat.id):
        buffer = await bot.download(file_id)
        assert buffer is not None
        try:
            text = await asyncio.to_thread(extract_text, buffer.read(), suffix)
        except EmptyDocumentError:
            await status.edit_text(texts.EMPTY_DOCUMENT)
            return
        except Exception:
            log.exception("failed to parse %s", file_name)
            await status.edit_text(texts.EMPTY_DOCUMENT)
            return
        result = await ingestor.prepare(
            text=text,
            file_name=file_name,
            owner_tg_id=message.from_user.id,
            owner_name=owner_name,
        )
        match result:
            case Duplicate(existing):
                await status.edit_text(
                    texts.DUPLICATE.format(
                        title=html.escape(existing.title), scope=existing.scope.label
                    )
                )
            case MixedPlaces(draft) | NeedsScope(draft):
                # Не вгадуємо й не ділимо документ самі — питаємо автора.
                question = (
                    texts.MIXED_PLACES if isinstance(result, MixedPlaces) else texts.ASK_SCOPE
                )
                await state.set_state(Upload.waiting_scope)
                await state.update_data(draft=draft.model_dump())
                await status.edit_text(
                    question.format(title=html.escape(draft.title)),
                    reply_markup=kb.scope_keyboard(),
                )
            case Ready(draft, scope):
                stored = await ingestor.store(draft, scope)
                await status.edit_text(
                    _stored_text(stored), reply_markup=kb.rescope_keyboard(stored.document.id)
                )


def _stored_text(stored: Stored, *, auto_scope: bool = True) -> str:
    doc = stored.document
    text = texts.STORED.format(
        name=html.escape(doc.owner_name or ""),
        title=html.escape(doc.title),
        scope=html.escape(doc.scope.label),
    )
    if stored.replaced:
        text += texts.REPLACED.format(title=html.escape(stored.replaced.title))
    if auto_scope:
        text += texts.WRONG_SCOPE_HINT
    return text


def _scope_from_key(place_id: str) -> Scope:
    return Scope.for_place(None if place_id == kb.FOREST_KEY else place_id)


@router.callback_query(kb.ChooseScope.filter(F.doc_id == 0))
async def on_scope_for_new(
    query: CallbackQuery,
    callback_data: kb.ChooseScope,
    state: FSMContext,
    ingestor: Ingestor,
) -> None:
    data = await state.get_data()
    if "draft" not in data or not isinstance(query.message, Message):
        await query.answer(texts.STALE_QUESTION, show_alert=True)
        return
    await query.answer()
    draft = Draft.model_validate(data["draft"])
    await state.clear()
    async with ChatActionSender.typing(bot=query.bot, chat_id=query.message.chat.id):  # type: ignore[arg-type]
        stored = await ingestor.store(draft, _scope_from_key(callback_data.place_id))
    await query.message.edit_text(_stored_text(stored, auto_scope=False))


@router.callback_query(kb.Rescope.filter())
async def on_rescope(query: CallbackQuery, callback_data: kb.Rescope, db: Database) -> None:
    doc = await db.get_document(callback_data.doc_id)
    if doc is None or not doc.is_active:
        await query.answer(texts.GONE, show_alert=True)
        return
    if doc.owner_tg_id != query.from_user.id:
        await query.answer(texts.NOT_YOURS, show_alert=True)
        return
    await query.answer()
    if isinstance(query.message, Message):
        await query.message.edit_reply_markup(reply_markup=kb.scope_keyboard(doc.id))


@router.callback_query(kb.ChooseScope.filter(F.doc_id != 0))
async def on_scope_for_existing(
    query: CallbackQuery, callback_data: kb.ChooseScope, db: Database
) -> None:
    doc = await db.get_document(callback_data.doc_id)
    if doc is None or not doc.is_active:
        await query.answer(texts.GONE, show_alert=True)
        return
    if doc.owner_tg_id != query.from_user.id:
        await query.answer(texts.NOT_YOURS, show_alert=True)
        return
    scope = _scope_from_key(callback_data.place_id)
    await db.update_scope(doc.id, scope)
    await query.answer()
    if isinstance(query.message, Message):
        await query.message.edit_text(
            texts.RESCOPED.format(title=html.escape(doc.title), scope=html.escape(scope.label))
        )


@router.callback_query(kb.Delete.filter())
async def on_delete(query: CallbackQuery, callback_data: kb.Delete, db: Database) -> None:
    doc = await db.get_document(callback_data.doc_id)
    if doc is None or not doc.is_active:
        await query.answer(texts.GONE, show_alert=True)
        return
    if doc.owner_tg_id != query.from_user.id:
        await query.answer(texts.NOT_YOURS, show_alert=True)
        return
    await query.answer()
    if not isinstance(query.message, Message):
        return
    title = html.escape(doc.title)
    if not callback_data.confirmed:
        await query.message.answer(
            texts.CONFIRM_DELETE.format(title=title),
            reply_markup=kb.confirm_delete_keyboard(doc.id),
        )
        return
    restored = await db.delete_document(doc.id)
    text = texts.DELETED.format(title=title)
    if restored:
        text += texts.RESTORED.format(title=html.escape(restored.title))
    await query.message.edit_text(text)


@router.callback_query(kb.CancelDelete.filter())
async def on_keep(query: CallbackQuery) -> None:
    await query.answer("Залишила 🌿")
    if isinstance(query.message, Message):
        await query.message.delete()


# --- питання ------------------------------------------------------------------


@router.message(F.text.startswith("/"))
async def on_unknown_command(message: Message) -> None:
    await message.answer(texts.UNKNOWN_COMMAND)


@router.message(F.text)
async def on_question(
    message: Message, state: FSMContext, bot: Bot, qa: QAService, memory: ConversationMemory
) -> None:
    assert message.from_user and message.text
    user_id = message.from_user.id
    async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
        reply = await qa.ask(message.text, memory.render(user_id))
    memory.add(user_id, message.text)
    await _send_reply(message, state, reply)


@router.callback_query(kb.AskPlace.filter())
async def on_place_chosen(
    query: CallbackQuery,
    callback_data: kb.AskPlace,
    state: FSMContext,
    qa: QAService,
    memory: ConversationMemory,
) -> None:
    question = (await state.get_data()).get("pending_question")
    if not question or not isinstance(query.message, Message):
        await query.answer(texts.STALE_QUESTION, show_alert=True)
        return
    await query.answer()
    await query.message.edit_reply_markup(reply_markup=None)
    async with ChatActionSender.typing(bot=query.bot, chat_id=query.message.chat.id):  # type: ignore[arg-type]
        reply = await qa.ask_for_place(
            question, callback_data.place_id, memory.render(query.from_user.id)
        )
    await _send_reply(query.message, state, reply)


async def _send_reply(message: Message, state: FSMContext, reply: Reply) -> None:
    if reply.needs_place:
        await state.update_data(pending_question=reply.question)
        await message.answer(reply.text, reply_markup=kb.places_keyboard())
        return
    await state.update_data(pending_question=None)
    for part in _split_message(reply.text):
        await message.answer(part)


def _split_message(text: str, limit: int = 4000) -> list[str]:
    parts = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        cut = cut if cut > 0 else limit
        parts.append(text[:cut])
        text = text[cut:].lstrip("\n")
    parts.append(text)
    return parts


@router.message()
async def on_other(message: Message) -> None:
    await message.answer(texts.PLEASE_TEXT)


# --- помилки ------------------------------------------------------------------


@router.error(ExceptionTypeFilter(LLMError))
async def on_llm_error(event: ErrorEvent) -> None:
    log.warning("LLM unavailable: %s", event.exception)
    await _notify(event, texts.THINKING_FAILED)


@router.error()
async def on_error(event: ErrorEvent) -> None:
    log.exception("unhandled error", exc_info=event.exception)
    await _notify(event, texts.UNEXPECTED_ERROR)


async def _notify(event: ErrorEvent, text: str) -> None:
    update = event.update
    target = update.message or (update.callback_query.message if update.callback_query else None)
    if isinstance(target, Message):
        await target.answer(text)
