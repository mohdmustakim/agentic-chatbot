
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from app.config import AGENT_MODEL, OLLAMA_BASE_URL

print("Model:", AGENT_MODEL)
print("Ollama URL:", OLLAMA_BASE_URL)

llm = ChatOllama(
    model=AGENT_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
    num_predict=128,
)

response = llm.invoke([
    HumanMessage(content="Explain RAG in 3 simple sentences.")
])

print("Response content:", repr(response.content))
print("Response metadata:", response.response_metadata)
