import re 
from langchain_core.prompts import ChatPromptTemplate

def split_sentences(text: str ) -> list[str]:
    """Split a note section into sentences (rule-based, so scoring is reproducible)."""
    parts = re.split (r"(?<=[.!?])\s+", text.strip()) #splitting into a single sentence 
    return [p.strip() for p in parts if p.strip()]

#GENERATOR PROMPT
GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Your role is an assistant for a GP. Your task is to write SOAP note from the consultation transcripts that will be provided. The SOAP rules include: \n"
        "1) Use only the information that is being provided in the transcript. Do not assume or derieve or add anything else apart from what is being discussed \n"
        "2) Subjective: patient's reported symptoms, history, medication taken, allergies."
        "3) Objective: examination findings, measurements and test results stated by the doctor.\n"
        "4) Assessment: the doctor's diagnosis or reasoning.\n"
        "5) Plan: treatment (with exact doses), tests, advice, safety-netting and follow-up.\n"
        "6) Write short and factual sentences. Also, keep the exact numbers and doses.\n"
        "7) If a certain detail is unknown, make sure to state it as 'Not Discussed'"
    ),

    (
        "human" , 
        "Transcript:\n{transcript}"
    ),
])


# FACT CHECKING PROMPT FOR THE JUDGE LLM
FACT_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict medical fact-checker. You will be provided with a consultation transcript and numbered sentences from one section of a clinical note.\n"
     "For EACH sentence decide whether the transcript supports it, based on these rules:\n"
     "1) Supported = the transcript states or clearly implies the information. Paraphrasing is allowed.\n"
     "2) Unsupported = fabricated information, a wrong detail (number, dose, duration, side), or a contradiction of the transcript.\n"
     "3) For unsupported sentences set error_type to one of: fabricated, wrong_detail, contradiction.\n"
     "4) Do NOT penalise missing information. Only judge what the sentence claims.\n"
     "5) Refer to each sentence by its number (sentence_id). Return exactly one result per sentence."),
    
    ("human",
     "Transcript:\n{transcript}\n\n"
     "Note section: {section}\n"
     "Sentences:\n{sentences}"),
])

# KEY FACT ALIGNMENT PROMPT FOR THE JUDGE LLM

KEY_FACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict medical reviewer. You are given numbered key facts and the numbered sentences of one section of a clinical note. "
     "For EACH key fact decide whether the section contains it.\n"
     "Rules:\n"
     "1) Covered = the section clearly states the fact. Paraphrasing is allowed.\n"
     "2) Specific details (numbers, doses, durations) must match, otherwise not covered.\n"
     "3) If covered, give the number of the matching sentence (matching_sentence_id); otherwise leave it null.\n"
     "4) Refer to each key fact by its number (key_fact_id). Return exactly one result per key fact."),

    ("human",
     "Note section: {section}\n"
     "Sentences:\n{sentences}\n\n"
     "Key facts:\n{key_facts}"),

])



# this is a helper function for the numbered lists 
def numbered(items: list[str]) -> str:
    return "\n".join(f"{i}. {x}" for i, x in enumerate(items, 1))
