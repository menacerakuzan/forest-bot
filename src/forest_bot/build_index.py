"""Оновлює data/archive_index.json — класифікує файли архіву, яких ще немає в індексі.

Запуск: docker compose run --rm -v ./data:/app/data bot python -m forest_bot.build_index
Працює в тимчасовій схемі PostgreSQL, тож дані бота не зачіпає.
"""

import asyncio
import logging
import uuid

from forest_bot.config import Settings
from forest_bot.db import Database
from forest_bot.ingest import Ingestor
from forest_bot.llm import LLM


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = Settings()  # type: ignore[call-arg]
    llm = LLM(
        settings.groq_api_key.get_secret_value(),
        fallback_model=settings.groq_model_fallback,
        max_retries=8,
    )
    schema = f"build_index_{uuid.uuid4().hex[:8]}"
    db = await Database.open(settings.database_url, schema=schema)
    try:
        ingestor = Ingestor(db, llm, model=settings.groq_model_answer)
        await ingestor.load_archive(settings.archive_dir, settings.archive_index_path)
        for doc in await db.active_documents():
            print(f"{doc.scope.label:34} {doc.title}")
    finally:
        await db.drop_schema(schema)
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
