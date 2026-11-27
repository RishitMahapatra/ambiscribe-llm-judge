import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return {c["id"]: c for c in json.load(f)}


def load_views(lang: str = "en") -> tuple[dict, dict]:
    """Returns (generator_view, judge_view), each keyed by consultation id."""
    gen = _load(DATA_DIR / lang / "generator_view.json")
    judge = _load(DATA_DIR / lang / "judge_view.json")
    assert gen.keys() == judge.keys(), "generator and judge views have different ids"
    return gen, judge


def format_transcript(turns: list[dict]) -> str:
    return "\n".join(f"{t['speaker']}: {t['text']}" for t in turns)