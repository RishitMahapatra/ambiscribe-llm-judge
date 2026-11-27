from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableParallel 
from .schemas import FactCheckResult, KeyFactAlignment

from .prompts import KEY_FACT_PROMPT , FACT_CHECK_PROMPT #importing the prompts 
from .prompts import split_sentences, numbered #importing the functions 

JUDGE_MODEL = "deepseek-r1:latest"
judge_llm = ChatOllama(model=JUDGE_MODEL, temperature=0, seed=42, num_ctx=8192, num_predict=3072, keep_alive="30m", reasoning=False)


fact_checking_chain = FACT_CHECK_PROMPT | judge_llm.with_structured_output(FactCheckResult)
key_alignment_chain = KEY_FACT_PROMPT | judge_llm.with_structured_output(KeyFactAlignment)
#parallel_runnables = RunnableParallel(fact_results = fact_checking_chain, align_results = key_alignment_chain)



def judge_section(transcript: str, section: str, text: str, key_facts: list[str]):
    inputs ={
        "transcript": transcript,
        "section": section,
        "sentences": numbered(split_sentences(text)),
        "key_facts": numbered(key_facts),
    }
    fact = fact_checking_chain.invoke(inputs)
    align = key_alignment_chain.invoke(inputs)
    return fact, align

    