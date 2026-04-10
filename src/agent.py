from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .store import EmbeddingStore


DEFAULT_SYSTEM_PROMPT = """<persona>
Ban la mot tro ly AI chuyen tra loi dua tren tri thuc noi bo da duoc truy xuat.
Ban noi chuyen ro rang, ngan gon, uu tien tinh chinh xac va tinh grounded.
</persona>

<rules>
1. Tra loi bang tieng Viet neu nguoi dung khong yeu cau ngon ngu khac.
2. Uu tien su dung context da retrieve duoc tu kho tri thuc.
3. Neu context chua du, hay noi ro phan nao ban chua chac thay vi tu suy doan.
4. Khi can, tom tat thong tin thanh cac y ngan gon, de doc.
</rules>

<rag_instruction>
- Hay xem phan Retrieved Context la nguon su that uu tien.
- Neu co nhieu chunk, tong hop chung thanh mot cau tra loi mach lac.
- Khong khang dinh nhung dieu khong co trong context.
</rag_instruction>
"""


def _load_default_system_prompt() -> str:
    prompt_path = Path(__file__).resolve().parent.parent / "system_prompt.txt"
    if prompt_path.exists():
        content = prompt_path.read_text(encoding="utf-8").strip()
        if content:
            return content
    return DEFAULT_SYSTEM_PROMPT.strip()


@dataclass
class ConversationTurn:
    user_message: str
    assistant_message: str
    sources: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentResponse:
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    prompt: str = ""


class KnowledgeBaseAgent:
    """
    Chatbot agent with a RAG-ready loop:
        1. Retrieve relevant chunks from the embedding store.
        2. Build a grounded prompt with history + retrieved context.
        3. Call the injected LLM function to generate the answer.

    The public ``answer`` method is kept for backwards compatibility with tests,
    while ``chat`` and ``chat_with_sources`` support multi-turn conversations.
    """

    def __init__(
        self,
        store: EmbeddingStore,
        llm_fn: Callable[[str], str],
        system_prompt: str | None = None,
        default_top_k: int = 3,
        max_history_turns: int = 6,
    ) -> None:
        self.store = store
        self.llm_fn = llm_fn
        self.system_prompt = (system_prompt or _load_default_system_prompt()).strip()
        self.default_top_k = max(1, default_top_k)
        self.max_history_turns = max(0, max_history_turns)
        self.history: list[ConversationTurn] = []

    def reset_conversation(self) -> None:
        """Clear all remembered chat turns."""
        self.history.clear()

    def retrieve(self, question: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Retrieve the most relevant chunks for a user question."""
        query = question.strip()
        if not query or self.store.get_collection_size() == 0:
            return []
        return self.store.search(query, top_k=top_k or self.default_top_k)

    def format_context(self, results: list[dict[str, Any]]) -> str:
        """Turn retrieved chunks into a readable prompt section."""
        if not results:
            return "Khong co context nao duoc retrieve."

        lines: list[str] = []
        for index, item in enumerate(results, start=1):
            metadata = item.get("metadata", {}) or {}
            source = metadata.get("source") or item.get("doc_id") or f"chunk_{index}"
            score = item.get("score", 0.0)
            content = str(item.get("content", "")).strip()
            lines.append(f"[Chunk {index}] source={source} score={score:.3f}")
            lines.append(f"Text: {content}")
        return "\n".join(lines)

    def build_prompt(self, question: str, results: list[dict[str, Any]]) -> str:
        """Build the full prompt for the model from system prompt + history + retrieval."""
        history_text = self._format_history()
        context_text = self.format_context(results)
        return (
            f"{self.system_prompt}\n\n"
            f"<conversation_history>\n{history_text}\n</conversation_history>\n\n"
            f"<retrieved_context>\n{context_text}\n</retrieved_context>\n\n"
            f"<user_question>\n{question.strip()}\n</user_question>\n\n"
            "<response_instruction>\n"
            "- Tra loi dua tren retrieved_context neu co.\n"
            "- Neu thieu du lieu, noi ro dieu gi chua du thong tin.\n"
            "- Neu co the, hay chi ra nguon bang ten file hoac source.\n"
            "</response_instruction>"
        )

    def answer_with_sources(self, question: str, top_k: int | None = None) -> AgentResponse:
        """Single-turn grounded QA without adding the turn to chat history."""
        return self._generate_response(question=question, top_k=top_k, remember=False)

    def chat_with_sources(self, message: str, top_k: int | None = None) -> AgentResponse:
        """Multi-turn chat that remembers the latest conversation turns."""
        return self._generate_response(question=message, top_k=top_k, remember=True)

    def answer(self, question: str, top_k: int = 3) -> str:
        """Backward-compatible single-turn answer API used by the tests."""
        return self.answer_with_sources(question, top_k=top_k).answer

    def chat(self, message: str, top_k: int | None = None) -> str:
        """Return only the answer text for an interactive chatbot turn."""
        return self.chat_with_sources(message, top_k=top_k).answer

    def _generate_response(self, question: str, top_k: int | None, remember: bool) -> AgentResponse:
        cleaned_question = question.strip()
        if not cleaned_question:
            return AgentResponse(answer="Moi ban nhap cau hoi cu the hon.", sources=[], prompt="")

        results = self.retrieve(cleaned_question, top_k=top_k)
        prompt = self.build_prompt(cleaned_question, results)
        answer = self.llm_fn(prompt).strip() or "Minh chua tao duoc cau tra loi phu hop."

        if remember:
            self._remember_turn(cleaned_question, answer, results)

        return AgentResponse(answer=answer, sources=results, prompt=prompt)

    def _remember_turn(self, user_message: str, assistant_message: str, sources: list[dict[str, Any]]) -> None:
        self.history.append(
            ConversationTurn(
                user_message=user_message,
                assistant_message=assistant_message,
                sources=[dict(item) for item in sources],
            )
        )
        if self.max_history_turns and len(self.history) > self.max_history_turns:
            self.history = self.history[-self.max_history_turns :]

    def _format_history(self) -> str:
        if not self.history:
            return "Chua co lich su hoi dap."

        lines: list[str] = []
        for index, turn in enumerate(self.history[-self.max_history_turns :], start=1):
            lines.append(f"Turn {index} - User: {turn.user_message}")
            lines.append(f"Turn {index} - Assistant: {turn.assistant_message}")
        return "\n".join(lines)


def create_agent(
    store: EmbeddingStore,
    llm_fn: Callable[[str], str],
    system_prompt: str | None = None,
    default_top_k: int = 3,
    max_history_turns: int = 6,
) -> KnowledgeBaseAgent:
    """Factory helper mirroring the Day04-style create_agent pattern."""
    return KnowledgeBaseAgent(
        store=store,
        llm_fn=llm_fn,
        system_prompt=system_prompt,
        default_top_k=default_top_k,
        max_history_turns=max_history_turns,
    )


def run_agent_loop(agent: KnowledgeBaseAgent, user_input: str, top_k: int | None = None) -> AgentResponse:
    """Run one chatbot turn and return answer + retrieved sources."""
    return agent.chat_with_sources(user_input, top_k=top_k)
