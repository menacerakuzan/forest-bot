from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Усі налаштування беруться зі змінних оточення (або з файлу .env)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    telegram_bot_token: SecretStr
    groq_api_key: SecretStr

    # Відповіді на питання — найсильніша модель безкоштовного рівня.
    groq_model_answer: str = "openai/gpt-oss-120b"
    # Службовий крок: відбір доречних документів за каталогом.
    groq_model_fast: str = "openai/gpt-oss-20b"
    # Запасна модель з окремим лімітом — коли основна впирається в rate limit.
    groq_model_fallback: str = "qwen/qwen3.8-27b"

    database_path: Path = Path("var/forest.db")
    archive_dir: Path = Path("data/archive")
    archive_index_path: Path = Path("data/archive_index.json")

    timezone: str = "Europe/Kyiv"
    log_level: str = "INFO"
    # Чи оновлювати назву/опис бота в Telegram під час старту.
    setup_bot_profile: bool = True
