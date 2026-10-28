from aiogram.filters.callback_data import CallbackData
from aiogram.types import InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

from forest_bot.db import Document
from forest_bot.places import GLOVES, PARTS


class AskPlace(CallbackData, prefix="ask"):
    place_id: str


class ChooseScope(CallbackData, prefix="scope"):
    place_id: str  # "forest" — весь ліс
    doc_id: int = 0  # 0 — новий документ, інакше зміна місця збереженого


class Rescope(CallbackData, prefix="rescope"):
    doc_id: int


class Delete(CallbackData, prefix="del"):
    doc_id: int
    confirmed: bool = False


class CancelDelete(CallbackData, prefix="keep"):
    pass


FOREST_KEY = "forest"


def places_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for place in (*GLOVES, *PARTS):
        builder.button(text=place.name, callback_data=AskPlace(place_id=place.id))
    builder.adjust(2)
    return builder.as_markup()


def scope_keyboard(doc_id: int = 0) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(
        text="🌳 Весь ліс", callback_data=ChooseScope(place_id=FOREST_KEY, doc_id=doc_id)
    )
    for place in (*PARTS, *GLOVES):
        builder.button(text=place.name, callback_data=ChooseScope(place_id=place.id, doc_id=doc_id))
    builder.adjust(1, 3, 2, 2, 2, 2)
    return builder.as_markup()


def rescope_keyboard(doc_id: int) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(text="📍 Змінити місце", callback_data=Rescope(doc_id=doc_id))
    return builder.as_markup()


def my_docs_keyboard(documents: list[Document]) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for doc in documents:
        builder.button(text=f"🗑 {doc.title[:48]}", callback_data=Delete(doc_id=doc.id))
    builder.adjust(1)
    return builder.as_markup()


def confirm_delete_keyboard(doc_id: int) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(text="Так, видалити", callback_data=Delete(doc_id=doc_id, confirmed=True))
    builder.button(text="Ні, залишити", callback_data=CancelDelete())
    return builder.as_markup()
