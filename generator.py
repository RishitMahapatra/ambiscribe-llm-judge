from langchain_ollama import ChatOllama
from schemas import SOAP

model = ChatOllama(model = "mistral:7b", temperature=0) # temp = 0 for reproducbility purposes 
generator = model.with_structured_output(SOAP)

result = generator.invoke("test transcript here")
print(result) 
