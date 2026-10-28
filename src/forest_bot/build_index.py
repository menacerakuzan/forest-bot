"""Оновлює data/archive_index.json — класифікує файли архіву, яких ще немає в індексі.

Запуск: docker compose run --rm -v ./data:/app/data bot python -m forest_bot.build_index
"""

import asyncio
import logging
import tempfile
from pathlib import Path

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
    with tempfile.TemporaryDirectory() as tmp:
        db = await Database.open(Path(tmp) / "index.db")
        try:
            ingestor = Ingestor(db, llm, model=settings.groq_model_answer)
            await ingestor.load_archive(settings.archive_dir, settings.archive_index_path)
            for doc in await db.active_documents():
                print(f"{doc.scope.label:34} {doc.title}")
        finally:
            await db.close()


if __name__ == "__main__":
    asyncio.run(main())
