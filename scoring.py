# scoring.py
from prompts import split_sentences


def score_section(text: str, n_key_facts: int, fact, align) -> dict:
    n = len(split_sentences(text))

    n_checks = len(fact.checks)
    faithfulness = sum(c.is_supported for c in fact.checks) / n_checks if n_checks else 0.0

    claimed = [m for m in align.matches if m.covered]
    valid = [m for m in claimed
             if isinstance(m.matching_sentence_id, int) and 1 <= m.matching_sentence_id <= n]
    completeness = len(valid) / n_key_facts if n_key_facts else 0.0
    conciseness = len({m.matching_sentence_id for m in valid}) / n if n else 0.0

    return {
        "faithfulness": round(faithfulness, 3),
        "completeness": round(completeness, 3),
        "conciseness": round(conciseness, 3),
        "n_sentences": n,
        "n_checks": n_checks,
        "unverified_covered": len(claimed) - len(valid),   # judge said covered but gave no valid sentence
    }