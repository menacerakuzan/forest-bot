import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramAPIError
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BotCommand

from forest_bot.bot import texts
from forest_bot.bot.handlers import router
from forest_bot.config import Settings
from forest_bot.db import Database
from forest_bot.ingest import Ingestor
from forest_bot.llm import LLM
from forest_bot.qa import ConversationMemory, QAService

log = logging.getLogger("forest_bot")

COMMANDS = [
    BotCommand(command="start", description="Почати"),
    BotCommand(command="help", description="Як користуватися"),
    BotCommand(command="my_docs", description="Мої документи"),
    BotCommand(command="name", description="Змінити ім'я автора"),
]


async def setup_profile(bot: Bot) -> None:
    """Назва, опис і команди бота. Telegram суворо лімітує ці виклики, тож помилки не фатальні."""
    me = await bot.get_me()
    calls = [
        bot.set_my_commands(COMMANDS),
        bot.set_my_description(texts.DESCRIPTION),
        bot.set_my_short_description(texts.SHORT_DESCRIPTION),
    ]
    if me.first_name != texts.BOT_NAME:
        calls.append(bot.set_my_name(texts.BOT_NAME))
    for call in calls:
        try:
            await call
        except TelegramAPIError as e:
            log.warning("could not update bot profile: %s", e)


async def main() -> None:
    settings = Settings()  # type: ignore[call-arg]
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    db = await Database.open(settings.database_url)
    llm = LLM(settings.groq_api_key.get_secret_value(), fallback_model=settings.groq_model_fallback)
    ingestor = Ingestor(db, llm, model=settings.groq_model_answer)
    await ingestor.load_archive(settings.archive_dir, settings.archive_index_path)
    qa = QAService(
        db,
        llm,
        answer_model=settings.groq_model_answer,
        fast_model=settings.groq_model_fast,
        timezone=settings.timezone,
    )

    bot = Bot(
        settings.telegram_bot_token.get_secret_value(),
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dispatcher = Dispatcher(storage=MemoryStorage())
    dispatcher.include_router(router)
    if settings.setup_bot_profile:
        await setup_profile(bot)

    log.info("bot is up")
    try:
        await dispatcher.start_polling(
            bot, db=db, ingestor=ingestor, qa=qa, memory=ConversationMemory()
        )
    finally:
        await db.close()
        await bot.session.close()


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
