from app.core.config import settings
from typing import Optional

def generate_answer(prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
    provider = settings.PROVIDER
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=settings.LLM_MODEL, temperature=temperature, convert_system_message_to_human=True)
        out = llm.invoke(prompt)
        return out.content if hasattr(out, "content") else str(out)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=temperature, max_tokens=max_tokens)
        out = llm.invoke(prompt)
        return out.content if hasattr(out, "content") else str(out)
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use gemini or openai.")
