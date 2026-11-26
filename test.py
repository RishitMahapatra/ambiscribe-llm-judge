from tqdm import tqdm
from langchain_ollama import ChatOllama

from loader import load_views, format_transcript
from schemas import SOAP
from prompts import GENERATOR_PROMPT, split_sentences
from judge import judge_section
from scoring import score_section

CONSULT_ID = "consult_01"
SECTIONS = ["subjective", "objective", "assessment", "plan"]


def by_id(items: list[str], i) -> str:
    """Safe 1-based lookup (the judge sometimes returns a bad id)."""
    return items[i - 1] if isinstance(i, int) and 1 <= i <= len(items) else "<invalid id>"


# ---------- load data ----------
gen_data, judge_data = load_views("en")
transcript = format_transcript(gen_data[CONSULT_ID]["transcript"])   # generator view only
key_facts = judge_data[CONSULT_ID]["key_facts"]                      # judge view only

# ---------- generate ----------
generator_llm = ChatOllama(model="mistral:7b", temperature=0, seed=42,
                           num_ctx=8192, num_predict=1024, keep_alive="30m")
generator = GENERATOR_PROMPT | generator_llm.with_structured_output(SOAP)

print("Generating SOAP note...")
note = generator.invoke({"transcript": transcript})
print(note.model_dump_json(indent=2))

# ---------- judge + score ----------
results = {}
for sec in tqdm(SECTIONS, desc="Judging sections"):
    text = getattr(note, sec)
    sentences = split_sentences(text)
    facts = key_facts[sec]

    fact, align = judge_section(transcript, sec.capitalize(), text, facts)

    tqdm.write(f"\n=== {sec.upper()} ===")
    for c in fact.checks:
        tqdm.write(f"[{'OK ' if c.is_supported else 'ERR'}] "
                   f"{by_id(sentences, c.sentence_id)[:70]} | {c.error_type or '-'}")
    for m in align.matches:
        tqdm.write(f"[{'COV ' if m.covered else 'MISS'}] "
                   f"{by_id(facts, m.key_fact_id)} -> note sentence {m.matching_sentence_id}")

    results[sec] = score_section(text, len(facts), fact, align)
    tqdm.write(str(results[sec]))

# ---------- overall ----------
overall = {k: round(sum(r[k] for r in results.values()) / len(results), 3)
           for k in ("faithfulness", "completeness", "conciseness")}
print("\nOVERALL:", overall)