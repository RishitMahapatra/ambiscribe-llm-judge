from pydantic import BaseModel, Field
from typing import Optional
from langchain_ollama import ChatOllama
from typing import TypedDict
from typing import Annotated



#OPTION 1 is to use BaseModel
class SOAP(BaseModel): # This is the SOAP generator output
    subjective: str = Field(description="The patient's reported symptoms, concerns, and medical history. Only the information stated in the transcript")
    objective: str = Field ( description= "Observable findings: physical exam results and measurements. Only information stated in the transcript.")
    assessment: str = Field (description="The clinician’s diagnostic reasoning based on the subjective and objective information")
    plan: str= Field(description="Follow-up steps: treatment, medication, tests, referrals and patient instructions included in the transcript")


#OPTION 2 is to use TypedDict
'''class TypedDict():
    Subjective: Annotated[str, "Write me the patient's reported symptoms, the concerns, the medical history and other sort of problems related to healthcare that is being reported in the transcript"]
    Objective: Annotated[str, "From the transcription, provide me the observable findings which will include physical examinations' results and other measureable data"]
    Assessment: Annotated[str, "Extract the clinician’s diagnostic reasoning based on the subjective and objective information"]
    Plan: Annotated[str, "Extract the Follow-up steps, such as treatment, tests, referrals, and patient instructions"]
'''

#JUDGE OUTPUT SCHEMA FOR FACT CHECKING 
# schemas.py: replace these two classes
class SentenceCheck(BaseModel):
    sentence_id: int = Field(description="Number of the sentence, as given in the list.")
    is_supported: bool = Field(description="True if the transcript supports this sentence.")
    error_type: Optional[str] = Field(default=None, description="If unsupported: fabricated, wrong_detail or contradiction. Otherwise null.")
    explanation: str = Field(description="One short reason for the verdict.")

    
class FactCheckResult(BaseModel): #this holds the list of sentence checks 
    checks: list[SentenceCheck]

#JUDGE OUTPUT FOR KEY FACT ALIGNMENT 
class KeyFactMatch(BaseModel):
    key_fact_id: int = Field(description="Number of the key fact, as given in the list.")
    covered: bool = Field(description="True if the note section contains this fact.")
    matching_sentence_id: Optional[int] = Field(default=None, description="Number of the note sentence that covers it, or null.")


class KeyFactAlignment(BaseModel): #this holds the list of keyfact matches. 
    matches: list[KeyFactMatch]
