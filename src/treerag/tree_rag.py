import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

try:
    from src.services.doc_processing_llm import DocProcessingLLMClient
except ImportError:
    from services.doc_processing_llm import DocProcessingLLMClient


logger = logging.getLogger(__name__)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _strip_numbered_prefix(line: str) -> str:
    # "1. foo" / "1) foo" / "- foo"
    return re.sub(r"^\s*(?:\d+[\.\)]\s+|-+\s+)\s*", "", line).strip()


def _extract_json_array(text: str) -> Optional[list[str]]:
    """
    Best-effort extraction of a JSON array from model output.
    Accepts either the whole string being JSON, or JSON embedded in text.
    """
    if not text:
        return None
    raw = text.strip()
    # Fast path: pure JSON
    try:
        val = json.loads(raw)
        if isinstance(val, list) and all(isinstance(x, str) for x in val):
            return [x.strip() for x in val if x and x.strip()]
    except Exception:
        pass
    # Embedded JSON array
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        snippet = raw[start : end + 1]
        try:
            val = json.loads(snippet)
            if isinstance(val, list) and all(isinstance(x, str) for x in val):
                return [x.strip() for x in val if x and x.strip()]
        except Exception:
            return None
    return None


def _format_chunks_for_prompt(chunks: list[dict], max_chunks: int) -> str:
    lines: list[str] = []
    for i, c in enumerate(chunks[:max_chunks]):
        meta = c.get("metadata") or {}
        published = meta.get("date_published") or "N/A"
        filename = meta.get("filename") or meta.get("source_doc_id") or "N/A"
        page = meta.get("page_number") or "N/A"
        section = meta.get("section_title") or "N/A"
        retriever = c.get("retriever") or "N/A"
        score = c.get("score")
        header = f"[{i+1}] source={filename} page={page} date={published} section={section} retriever={retriever} score={score}"
        content = (c.get("page_content") or "").strip()
        lines.append(f"{header}\n{content}")
    return "\n\n".join(lines)


@dataclass
class TreeRagNode:
    id: str
    question: str
    retrieved_chunks: list[dict]
    summary: str
    is_sufficient: bool
    children: list["TreeRagNode"] = field(default_factory=list)
    combined_summary: Optional[str] = None


class TreeRagEngine:
    """
    Tree-of-thought style RAG orchestrator.
    - Uses existing retrieval stack via RAGManager.get_retriever(collection_name)
    - Uses centralized LLM service llm/complete calls configured via config/*
    """

    def __init__(
        self,
        *,
        config: dict,
        rag_manager: Any,
        chat_service: Any,
        max_workers: int,
    ):
        self._config = config
        self._rag_manager = rag_manager
        self._chat_service = chat_service
        self._max_workers = max_workers

        self._default_model: str = config.get("treerag_llm_model_name") or config.get("llm_model_name")
        self._planner_model: str = config.get("treerag_planner_model") or self._default_model
        self._answer_model: str = config.get("treerag_answer_model") or self._default_model

        self._client = DocProcessingLLMClient.from_config(config, provider_key="treerag_llm_provider")

    def _chat(self, *, model: str, messages: list[dict], temperature: float = 0.2) -> str:
        resp = self._client.complete(
            model=model,
            messages=messages,
        )
        return str(resp.get("content", "")).strip()

    def _retrieve(self, *, collection_name: str, question: str, session_id: str) -> list[dict]:
        retriever = self._rag_manager.get_retriever(collection_name)
        use_hyde = bool(self._config.get("treerag_use_hyde", False))
        hyde_chunks: list[str] = []
        if use_hyde:
            chat_manager = self._chat_service.get_or_create_chat_manager(session_id, collection_name)
            hyde_chunks = chat_manager.generate_hypo_chunks(question)
        return retriever.invoke(question, hyde_chunks)

    def _summarize(self, *, question: str, chunks: list[dict], max_chunks: int) -> str:
        context = _format_chunks_for_prompt(chunks, max_chunks=max_chunks)
        prompt = (
            "You are a careful RAG assistant.\n"
            "Task: extract only the information that helps answer the user's question.\n"
            "If the context does not contain enough information, say what is missing.\n\n"
            f"Question:\n{question}\n\n"
            f"Context chunks:\n{context}\n\n"
            "Write a concise summary (<= 8 sentences)."
        )
        return self._chat(
            model=self._planner_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

    def _check_sufficiency(self, *, question: str, summary: str) -> tuple[bool, str]:
        prompt = (
            "Determine whether the summary is sufficient to fully answer the question.\n"
            "Respond ONLY as JSON:\n"
            '{"sufficient": true|false, "reason": "short reason"}\n\n'
            f"Question:\n{question}\n\n"
            f"Summary:\n{summary}\n"
        )
        raw = self._chat(model=self._planner_model, messages=[{"role": "user", "content": prompt}], temperature=0.0)
        try:
            data = json.loads(raw)
            return bool(data.get("sufficient", False)), str(data.get("reason", "")).strip()
        except Exception:
            # Fallback: heuristic
            upper = raw.strip().upper()
            if "TRUE" in upper or "YES" in upper:
                return True, raw
            return False, raw

    def _generate_followups(self, *, original_question: str, summary: str, num_questions: int) -> list[str]:
        prompt = (
            "Generate follow-up questions that would help answer the original question.\n"
            "Respond ONLY as a JSON array of strings, with exactly "
            f"{num_questions} items unless fewer are truly necessary.\n\n"
            f"Original question:\n{original_question}\n\n"
            f"Current summary:\n{summary}\n"
        )
        raw = self._chat(model=self._planner_model, messages=[{"role": "user", "content": prompt}], temperature=0.3)
        arr = _extract_json_array(raw)
        if arr is not None:
            return arr[:num_questions]
        # Fallback: parse lines
        lines = [_strip_numbered_prefix(x) for x in raw.splitlines() if x.strip()]
        return [x for x in lines if x][:num_questions]

    def _combine(self, *, question: str, root_summary: str, child_summaries: list[tuple[str, str]]) -> str:
        extra = "\n\n".join([f"Follow-up question: {q}\nSummary: {s}" for q, s in child_summaries])
        prompt = (
            "Combine the root summary with follow-up summaries into a single coherent summary.\n"
            "Do not invent facts not supported by the provided summaries.\n\n"
            f"Main question:\n{question}\n\n"
            f"Root summary:\n{root_summary}\n\n"
            f"Follow-up summaries:\n{extra}\n\n"
            "Write a concise combined summary (<= 10 sentences)."
        )
        return self._chat(model=self._planner_model, messages=[{"role": "user", "content": prompt}], temperature=0.2)

    def _final_answer(self, *, question: str, combined_summary: str) -> str:
        prompt = (
            "Answer the question using the information below.\n"
            "If the information is insufficient, say what is missing and answer as best as possible.\n\n"
            f"Question:\n{question}\n\n"
            f"Information:\n{combined_summary}\n\n"
            "Write a final answer (<= 10 sentences)."
        )
        return self._chat(model=self._answer_model, messages=[{"role": "user", "content": prompt}], temperature=0.2)

    def build_tree(
        self,
        *,
        question: str,
        session_id: str,
        collection_name: str,
        max_depth: int,
        current_depth: int = 0,
    ) -> TreeRagNode:
        retrieve_max_chunks = _safe_int(self._config.get("treerag_retrieve_max_chunks", 40), 40)
        branching_factor = _safe_int(self._config.get("treerag_branching_factor", 3), 3)

        chunks = self._retrieve(collection_name=collection_name, question=question, session_id=session_id)
        summary = self._summarize(question=question, chunks=chunks, max_chunks=retrieve_max_chunks)
        sufficient, _reason = self._check_sufficiency(question=question, summary=summary)

        node = TreeRagNode(
            id=str(uuid.uuid4()),
            question=question,
            retrieved_chunks=chunks,
            summary=summary,
            is_sufficient=sufficient,
        )

        if sufficient or current_depth >= max_depth:
            node.combined_summary = summary
            return node

        followups = self._generate_followups(
            original_question=question,
            summary=summary,
            num_questions=branching_factor,
        )
        if not followups:
            node.combined_summary = summary
            return node

        from concurrent.futures import ThreadPoolExecutor, as_completed

        child_nodes: list[TreeRagNode] = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = [
                executor.submit(
                    self.build_tree,
                    question=fq,
                    session_id=session_id,
                    collection_name=collection_name,
                    max_depth=max_depth,
                    current_depth=current_depth + 1,
                )
                for fq in followups
            ]
            for fut in as_completed(futures):
                try:
                    child_nodes.append(fut.result())
                except Exception as e:
                    logger.exception("TreeRAG child failed: %s", e)

        node.children = child_nodes
        node.combined_summary = self._combine(
            question=question,
            root_summary=summary,
            child_summaries=[(c.question, c.combined_summary or c.summary) for c in child_nodes],
        )
        return node

    def run(
        self,
        *,
        question: str,
        session_id: str,
        collection_name: str,
        max_depth: int,
    ) -> tuple[str, TreeRagNode]:
        root = self.build_tree(
            question=question,
            session_id=session_id,
            collection_name=collection_name,
            max_depth=max_depth,
            current_depth=0,
        )
        combined = root.combined_summary or root.summary
        answer = self._final_answer(question=question, combined_summary=combined)
        return answer, root


def node_to_tree_dict(node: TreeRagNode) -> dict:
    return {
        "id": node.id,
        "question": node.question,
        "is_sufficient": node.is_sufficient,
        "children": [node_to_tree_dict(c) for c in node.children],
    }


def find_node_details(node: TreeRagNode, node_id: str) -> Optional[dict]:
    if node.id == node_id:
        return {
            "id": node.id,
            "question": node.question,
            "summary": node.summary,
            "combined_summary": node.combined_summary,
            "is_sufficient": node.is_sufficient,
            "retrieved_chunks": node.retrieved_chunks,
        }
    for child in node.children:
        hit = find_node_details(child, node_id)
        if hit:
            return hit
    return None


@dataclass
class TreeRagSession:
    updated_at: datetime
    root: TreeRagNode

