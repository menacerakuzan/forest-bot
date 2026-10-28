"""Тонка обгортка над Groq: JSON-відповіді з валідацією схеми та запасною моделлю.

Використовуємо JSON Object Mode, а не strict json_schema: на gpt-oss/qwen у Groq строге
декодування нестабільне (порожні списки, «Failed to validate JSON»), тоді як JSON-режим із
схемою в промпті і перевіркою pydantic дає стабільний результат.
"""

import json
import logging
from typing import Any, TypeVar

from groq import APIConnectionError, APIStatusError, AsyncGroq, RateLimitError
from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """LLM недоступна або не повернула коректну відповідь."""


def _is_reasoning_model(model: str) -> bool:
    return model.startswith("openai/gpt-oss")


def _describe(info: dict[str, Any]) -> str:
    if "enum" in info:
        return " | ".join(json.dumps(v, ensure_ascii=False) for v in info["enum"])
    if "anyOf" in info:
        return " | ".join(_describe(option) for option in info["anyOf"])
    if info.get("type") == "array":
        return f"array of {_describe(info.get('items', {}))}"
    return str(info.get("type", "any"))


def _schema_hint(schema: type[BaseModel]) -> str:
    """Стисла схема полів для промпту: {"found": boolean, "level": "forest" | "part", ...}."""
    properties = schema.model_json_schema()["properties"]
    return (
        "{" + ", ".join(f'"{name}": {_describe(info)}' for name, info in properties.items()) + "}"
    )


class LLM:
    def __init__(
        self, api_key: str, *, fallback_model: str | None = None, max_retries: int = 1
    ) -> None:
        # Мало повторів: при вичерпаному ліміті швидше перейти на запасну модель,
        # ніж змушувати звіра чекати.
        self._client = AsyncGroq(api_key=api_key, max_retries=max_retries, timeout=60)
        self._fallback_model = fallback_model

    async def structured(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: type[T],
        reasoning_effort: str = "low",
    ) -> T:
        """Повертає відповідь моделі як екземпляр schema; при збої пробує запасну модель."""
        models = [model]
        if self._fallback_model and self._fallback_model != model:
            models.append(self._fallback_model)
        last_error: Exception | None = None
        for candidate in models:
            try:
                return await self._call(candidate, system, user, schema, reasoning_effort)
            except (RateLimitError, APIConnectionError, ValidationError, ValueError) as e:
                last_error = e
            except APIStatusError as e:
                # 400 — модель не вклалася в JSON, 413 — завеликий запит, 5xx — збій Groq.
                if e.status_code not in (400, 413) and e.status_code < 500:
                    raise LLMError(str(e)) from e
                last_error = e
            log.warning("LLM %s failed for %s: %s", candidate, schema.__name__, last_error)
        raise LLMError(str(last_error)) from last_error

    async def _call(
        self, model: str, system: str, user: str, schema: type[T], reasoning_effort: str
    ) -> T:
        request: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": f"{system}\n\nВідповідай лише одним JSON-об'єктом з полями: "
                    f"{_schema_hint(schema)}",
                },
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        if _is_reasoning_model(model):
            request["reasoning_effort"] = reasoning_effort
        response = await self._client.chat.completions.create(**request)
        content = response.choices[0].message.content
        if not content:
            raise ValueError("порожня відповідь моделі")
        return schema.model_validate_json(content)
