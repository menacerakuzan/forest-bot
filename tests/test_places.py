import pytest

from forest_bot.places import FOREST, Level, Scope, applicable_scopes, detect_places


def test_glove_gets_its_part_and_whole_forest() -> None:
    assert applicable_scopes("dub") == [
        Scope(Level.GLOVE, "dub"),
        Scope(Level.PART, "deciduous"),
        FOREST,
    ]


def test_verba_belongs_to_two_parts() -> None:
    assert applicable_scopes("verba") == [
        Scope(Level.GLOVE, "verba"),
        Scope(Level.PART, "deciduous"),
        Scope(Level.PART, "river"),
        FOREST,
    ]


def test_uzlissya_belongs_to_no_part() -> None:
    assert applicable_scopes("uzlissya") == [Scope(Level.GLOVE, "uzlissya"), FOREST]


def test_part_and_forest() -> None:
    assert applicable_scopes("river") == [Scope(Level.PART, "river"), FOREST]
    assert applicable_scopes(None) == [FOREST]


def test_priority_prefers_specific() -> None:
    assert Scope(Level.GLOVE, "dub").priority < Scope(Level.PART, "deciduous").priority
    assert Scope(Level.PART, "deciduous").priority < FOREST.priority


@pytest.mark.parametrize(
    ("level", "place"),
    [(Level.FOREST, "dub"), (Level.GLOVE, None), (Level.GLOVE, "river"), (Level.PART, "dub")],
)
def test_scope_rejects_inconsistent_values(level: Level, place: str | None) -> None:
    with pytest.raises(ValueError):
        Scope(level, place)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Скільки внесок під дубом?", ["dub"]),
        ("а під вербою?", ["verba"]),
        ("Хто голова ОСМД у соснах?", ["sosny"]),
        ("Що їсти в хвойному лісі?", ["conifer"]),
        ("Коли бобри відкривають греблю?", ["greblya"]),
        ("Що є під вербою біля річки?", ["verba"]),
        ("Сколько взнос под ёлкой?", ["yalyna"]),
        ("Кто дежурит ночью у сосен?", ["sosny"]),
        ("Порівняй внески біля мурашника і на узліссі", ["murashnyk", "uzlissya"]),
        ("Небезпеки біля річки", ["river"]),
        ("Що робити при пожежі?", []),
        ("Як не заблукати серед сосен?", []),
    ],
)
def test_detect_places(question: str, expected: list[str]) -> None:
    assert detect_places(question) == expected
