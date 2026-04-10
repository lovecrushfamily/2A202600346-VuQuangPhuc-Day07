from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agent import AgentResponse, create_agent, run_agent_loop
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

SAMPLE_FILES = [
    "data/python_intro.txt",
    "data/vector_store_notes.md",
    "data/rag_system_design.md",
    "data/customer_support_playbook.txt",
    "data/chunking_experiment_report.md",
    "data/vi_retrieval_notes.md",
]

DEFAULT_CHAT_MODEL = "qwen/qwen3.6-plus:free"


def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load text documents from .md and .txt files."""
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)
        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path}")
            continue
        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        documents.append(
            Document(
                id=path.stem,
                content=path.read_text(encoding="utf-8"),
                metadata={"source": str(path), "extension": path.suffix.lower()},
            )
        )

    return documents


def demo_llm(prompt: str) -> str:
    """
    Deterministic demo responder used when no real chat model is connected yet.

    It keeps the agent loop usable for classroom demos and makes retrieved
    sources visible, while leaving a clean seam for plugging in a real LLM later.
    """

    question = _extract_tag(prompt, "user_question")
    context = _extract_tag(prompt, "retrieved_context")
    snippets = _extract_context_snippets(context)

    if not snippets:
        return (
            "Minh chua retrieve duoc context phu hop de tra loi chac chan. "
            "Ban co the thu hoi cu the hon hoac nap them tai lieu cho kho tri thuc."
        )

    answer_lines = [
        f"Tra loi tam dua tren context vua retrieve cho cau hoi: {question}",
        f"- Tom tat nhanh: {snippets[0]}",
    ]
    if len(snippets) > 1:
        answer_lines.append(f"- Bo sung: {snippets[1]}")
    answer_lines.append("- Ghi chu: hien tai day la demo_llm, ban co the thay bang OpenAI/Claude sau.")
    return "\n".join(answer_lines)


def build_chat_llm():
    """
    Return a callable LLM function.

    Priority:
    1. OpenRouter via API key, matching the Lab4 setup style.
    2. Fallback to the classroom-safe demo_llm.
    """

    load_dotenv(override=False)
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    model_name = os.getenv("MODEL_NAME", DEFAULT_CHAT_MODEL).strip() or DEFAULT_CHAT_MODEL

    if openrouter_api_key:
        try:
            from langchain_openrouter import ChatOpenRouter

            client = ChatOpenRouter(model=model_name)

            def _openrouter_llm(prompt: str) -> str:
                response = client.invoke(prompt)
                content = getattr(response, "content", response)
                if isinstance(content, list):
                    return "\n".join(str(item) for item in content)
                return str(content)

            setattr(_openrouter_llm, "_backend_name", f"openrouter:{model_name}")
            return _openrouter_llm
        except Exception:
            pass

    setattr(demo_llm, "_backend_name", "demo_llm")
    return demo_llm


def _extract_tag(prompt: str, tag_name: str) -> str:
    start_token = f"<{tag_name}>"
    end_token = f"</{tag_name}>"
    start = prompt.find(start_token)
    end = prompt.find(end_token)
    if start == -1 or end == -1 or end <= start:
        return ""
    return prompt[start + len(start_token) : end].strip()


def _extract_context_snippets(context_block: str) -> list[str]:
    snippets: list[str] = []
    for line in context_block.splitlines():
        stripped = line.strip()
        if stripped.startswith("Text:"):
            snippets.append(stripped.removeprefix("Text:").strip())
    return snippets


def get_embedder():
    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()

    if provider == "local":
        try:
            return LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            return _mock_embed

    if provider == "openai":
        try:
            return OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            return _mock_embed

    return _mock_embed


def build_demo_store(sample_files: list[str] | None = None) -> EmbeddingStore:
    files = sample_files or SAMPLE_FILES
    documents = load_documents_from_files(files)
    store = EmbeddingStore(collection_name="day07_chatbot_store", embedding_fn=get_embedder())
    store.add_documents(documents)
    return store


def print_sources(response: AgentResponse) -> None:
    if not response.sources:
        print("Sources: khong co chunk nao duoc retrieve.")
        return

    print("Sources:")
    for index, item in enumerate(response.sources, start=1):
        metadata = item.get("metadata", {}) or {}
        source = metadata.get("source") or item.get("doc_id") or f"chunk_{index}"
        score = item.get("score", 0.0)
        print(f"  {index}. score={score:.3f} source={source}")


def run_single_turn(question: str, sample_files: list[str] | None = None) -> int:
    store = build_demo_store(sample_files=sample_files)
    if store.get_collection_size() == 0:
        print("Khong co tai lieu hop le de tao knowledge base.")
        return 1

    llm_fn = build_chat_llm()
    agent = create_agent(store=store, llm_fn=llm_fn)
    response = run_agent_loop(agent, question)
    print(f"LLM backend: {getattr(llm_fn, '_backend_name', llm_fn.__class__.__name__)}")
    print(f"Question: {question}")
    print(f"\nAgent:\n{response.answer}")
    print()
    print_sources(response)
    return 0


def run_chat_cli(sample_files: list[str] | None = None) -> int:
    files = sample_files or SAMPLE_FILES
    store = build_demo_store(sample_files=files)
    if store.get_collection_size() == 0:
        print("Khong co tai lieu hop le de tao knowledge base.")
        print("Hay kiem tra lai cac file sau:")
        for file_path in files:
            print(f"  - {file_path}")
        return 1

    llm_fn = build_chat_llm()
    agent = create_agent(store=store, llm_fn=llm_fn)

    print("=" * 60)
    print("Day07 RAG Chatbot")
    print("Lenh ho tro: /reset, /sources, quit")
    print(f"Knowledge base size: {store.get_collection_size()} documents")
    print(f"LLM backend: {getattr(llm_fn, '_backend_name', llm_fn.__class__.__name__)}")
    print("=" * 60)

    last_response: AgentResponse | None = None

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break
        if user_input == "/reset":
            agent.reset_conversation()
            last_response = None
            print("Conversation da duoc reset.")
            continue
        if user_input == "/sources":
            if last_response is None:
                print("Chua co luot tra loi nao de hien source.")
            else:
                print_sources(last_response)
            continue

        last_response = run_agent_loop(agent, user_input)
        print(f"\nAgent:\n{last_response.answer}")
        print()
        print_sources(last_response)

    return 0


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
    if question:
        return run_single_turn(question=question)
    return run_chat_cli()


if __name__ == "__main__":
    raise SystemExit(main())
