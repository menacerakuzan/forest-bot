"""Географія лісу: рукавички, частини лісу й те, які поради до якого місця застосовні."""

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class Level(StrEnum):
    FOREST = "forest"
    PART = "part"
    GLOVE = "glove"


@dataclass(frozen=True)
class Place:
    id: str
    level: Level
    name: str
    """Як показувати на кнопках і в заголовках."""
    label: str
    """Як вписувати в речення («рукавичка «Під дубом»», «хвойний ліс»)."""


PARTS: tuple[Place, ...] = (
    Place("conifer", Level.PART, "Хвойний ліс", "хвойний ліс"),
    Place("deciduous", Level.PART, "Листяний ліс", "листяний ліс"),
    Place("river", Level.PART, "Біля річки", "біля річки"),
)

GLOVES: tuple[Place, ...] = tuple(
    Place(pid, Level.GLOVE, name, f"рукавичка «{name}»")
    for pid, name in (
        ("yalyna", "Під ялиною"),
        ("sosny", "У соснах"),
        ("murashnyk", "Біля мурашника"),
        ("dub", "Під дубом"),
        ("lishchyna", "У ліщині"),
        ("verba", "Під вербою"),
        ("greblya", "Біля бобрової греблі"),
        ("uzlissya", "На узліссі"),
    )
)

# За «Картою лісу»: Під вербою — одразу у двох частинах, На узліссі — у жодній.
GLOVE_PARTS: dict[str, tuple[str, ...]] = {
    "yalyna": ("conifer",),
    "sosny": ("conifer",),
    "murashnyk": ("conifer",),
    "dub": ("deciduous",),
    "lishchyna": ("deciduous",),
    "verba": ("deciduous", "river"),
    "greblya": ("river",),
    "uzlissya": (),
}

PLACES: dict[str, Place] = {p.id: p for p in (*PARTS, *GLOVES)}

PlaceId = Literal[
    "conifer",
    "deciduous",
    "river",
    "yalyna",
    "sosny",
    "murashnyk",
    "dub",
    "lishchyna",
    "verba",
    "greblya",
    "uzlissya",
]


@dataclass(frozen=True)
class Scope:
    """Рівень, на якому діє порада: весь ліс, частина лісу або рукавичка."""

    level: Level
    place_id: str | None = None

    def __post_init__(self) -> None:
        if self.level is Level.FOREST:
            if self.place_id is not None:
                raise ValueError("порада для всього лісу не має місця")
        elif self.place_id is None or PLACES[self.place_id].level is not self.level:
            raise ValueError(f"місце {self.place_id!r} не відповідає рівню {self.level}")

    @property
    def label(self) -> str:
        return "весь ліс" if self.place_id is None else PLACES[self.place_id].label

    @property
    def priority(self) -> int:
        """Менше число — конкретніша порада, вона важливіша."""
        return {Level.GLOVE: 0, Level.PART: 1, Level.FOREST: 2}[self.level]

    @classmethod
    def for_place(cls, place_id: str | None) -> "Scope":
        if place_id is None:
            return cls(Level.FOREST)
        return cls(PLACES[place_id].level, place_id)


FOREST = Scope(Level.FOREST)


def applicable_scopes(place_id: str | None) -> list[Scope]:
    """Усі рівні порад, що стосуються місця, від найконкретнішого до загального."""
    if place_id is None:
        return [FOREST]
    place = PLACES[place_id]
    if place.level is Level.PART:
        return [Scope(Level.PART, place_id), FOREST]
    parts = [Scope(Level.PART, part) for part in GLOVE_PARTS[place_id]]
    return [Scope(Level.GLOVE, place_id), *parts, FOREST]


# Як звірі називають місця (українською й російською, у різних відмінках).
_PLACE_PATTERNS: dict[str, str] = {
    "yalyna": r"ялин|під\s+ялинк|под\s+[её]л|[её]лк|[её]лью",
    "sosny": r"[ув]\s+сосн|під\s+сосн|под\s+сосн",
    "murashnyk": r"мурашник|муравейник",
    "dub": r"\bдуб",
    "lishchyna": r"ліщин|лещин|орешник",
    "verba": r"верб|\bив(ой|ы|а|е|у)\b|под\s+ив",
    "greblya": r"гребл|бобров|плотин",
    "uzlissya": r"узлісс|опушк",
    "conifer": r"хвойн",
    "deciduous": r"листян|лиственн",
    "river": r"річк|річц|\bрек[аиеу]|речк",
}
_COMPILED = {pid: re.compile(p, re.IGNORECASE) for pid, p in _PLACE_PATTERNS.items()}


def detect_places(text: str) -> list[str]:
    """Місця, явно названі в тексті. Частину лісу відкидаємо, якщо названо її рукавичку."""
    text = text.replace("’", "'").replace("ʼ", "'")
    found = [pid for pid, pattern in _COMPILED.items() if pattern.search(text)]
    gloves = [pid for pid in found if PLACES[pid].level is Level.GLOVE]
    covered = {part for glove in gloves for part in GLOVE_PARTS[glove]}
    # «Біля бобрової греблі» сама по собі біля річки — не дублюємо частину.
    return [pid for pid in found if pid not in covered]


def geography_description() -> str:
    """Опис лісу для промптів LLM."""
    lines = ["Частини лісу (id = назва: рукавички в ній):"]
    for part in PARTS:
        members = ", ".join(PLACES[g].name for g, parts in GLOVE_PARTS.items() if part.id in parts)
        lines.append(f"- {part.id} = {part.name}: {members}")
    lines.append("Рукавички (id = назва):")
    lines.extend(f"- {g.id} = {g.name}" for g in GLOVES)
    lines.append(
        "«Під вербою» належить одразу до листяного лісу і до «біля річки». "
        "«На узліссі» не належить до жодної частини."
    )
    return "\n".join(lines)
