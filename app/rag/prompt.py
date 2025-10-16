from langchain.schema import Document

MAX_CHARS = 1200  
SYSTEM_INSTRUCTION = (
    "You are a helpful assistant that MUST answer ONLY using the provided context. "
    "Cite facts from the context; do not fabricate."
    "If the answer is not present, say so briefly."
)

def build_prompt(context_docs, question: str) -> str:
    context_texts = []
    for d in context_docs:
        txt = (d.page_content or "")[:MAX_CHARS]
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        ref = f"(source: {src}, page: {page})" if page is not None else f"(source: {src})"
        context_texts.append(f"{ref}\n{txt}")
    joined_context = "\n\n---\n\n".join(context_texts)
    return f"""{SYSTEM_INSTRUCTION}

Context:
{joined_context}

Question: {question}

Answer in the same language as the question."""
    return prompt
