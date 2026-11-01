import os
from dotenv import load_dotenv

load_dotenv()

# LLM provider: "groq" (cloud) or "ollama" (local)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def get_llm(provider=None, model=None):
    provider = (provider or LLM_PROVIDER).lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model or OLLAMA_MODEL,
                          base_url=OLLAMA_BASE_URL,
                          format="json",     # constrain output to valid JSON
                          temperature=0,
                          num_ctx=8192,      # Ollama's default context is too small for a full resume
                          repeat_penalty=1.1,  # small models otherwise get stuck repeating one list item
                          num_predict=4096)    # hard stop for any runaway generation that still slips through

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"),
                        model=model or GROQ_MODEL,
                        # Force syntactically valid JSON so the Pydantic parsers don't choke
                        model_kwargs={"response_format": {"type": "json_object"}},
                        max_retries=8)

    raise ValueError(f"Unknown LLM_PROVIDER '{provider}' (expected 'groq' or 'ollama')")


# Constrain the LLM's output to a Pydantic model's JSON schema where the provider supports it.
# Small local models otherwise tend to echo the schema back instead of filling it in.
def with_schema(llm, model_cls):
    from langchain_ollama import ChatOllama
    if isinstance(llm, ChatOllama):
        return llm.bind(format=model_cls.model_json_schema())
    return llm
