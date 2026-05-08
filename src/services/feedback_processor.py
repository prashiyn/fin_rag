"""
Process feedback records: classify questions, match to frequent QA, write to feedback_question_aliases.
Runs periodically from the server. Categories and descriptions are loaded from config/feedback_categories.json.
"""
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from database import get_session
from models import Feedback, FeedbackQuestionAlias, FrequentQAPair
try:
    from src.services.doc_processing_llm import DocProcessingLLMClient
except ImportError:
    from services.doc_processing_llm import DocProcessingLLMClient

logger = logging.getLogger(__name__)

DEFAULT_CATEGORIES_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "feedback_categories.json"


def load_categories(categories_path: str | Path | None = None) -> tuple[list[str], dict[str, str]]:
    """Load categories and category_descriptions from JSON. Returns (categories_list, category_descriptions_dict)."""
    path = Path(categories_path) if categories_path else DEFAULT_CATEGORIES_PATH
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    categories = data.get("categories", [])
    descriptions = data.get("category_descriptions", {})
    return categories, descriptions


def get_last_processed_id(file_path: str | Path) -> int:
    """Read last processed feedback ID from file."""
    path = Path(file_path)
    if not path.exists():
        return 0
    try:
        return int(path.read_text().strip())
    except (ValueError, OSError):
        return 0


def save_last_processed_id(file_path: str | Path, last_id: int) -> None:
    """Write last processed feedback ID to file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(last_id), encoding="utf-8")


def _get_classifier_client(config: dict) -> DocProcessingLLMClient | None:
    """Build centralized llm client for classifier model."""
    if not config.get("llm_service_base_url"):
        return None
    try:
        return DocProcessingLLMClient.from_config(config, provider_key="feedback_classifier_provider")
    except Exception as e:
        logger.warning("Failed to build classifier client: %s", e)
        return None


def classify_question(
    question: str,
    is_rag: int,
    categories: list[str],
    category_descriptions: dict[str, str],
    client: DocProcessingLLMClient | None,
    model: str,
    retries: int = 3,
) -> str:
    """Classify question into one of the categories using LLM. Returns 'non_rag' if is_rag==0 or client is None."""
    if is_rag == 0 or not question:
        return "non_rag"
    if not client or not model:
        return "non_rag"
    prompt_desc = "\n".join(
        f"{i + 1}. {cat}: {category_descriptions.get(cat, '')}" for i, cat in enumerate(categories)
    )
    prompt = f"""Classify the following question into exactly one of these categories:

{prompt_desc}

Question: {question}

Return ONLY the category name without any explanation or additional text. For example, if the question is about headquarters, just return "Company_Basics_Governance". Do not include numbers, punctuation, or anything else."""
    for attempt in range(retries):
        try:
            response = client.complete(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that classifies questions into predefined categories."},
                    {"role": "user", "content": prompt},
                ],
            )
            category = str(response.get("content", "")).strip()
            if category in categories:
                return category
            for valid in categories:
                if valid.lower() in category.lower():
                    return valid
            logger.warning("Classifier returned invalid category %r for question; defaulting to non_rag", category)
            return "non_rag"
        except Exception as e:
            if attempt < retries - 1:
                logger.warning("Classifier API error (retrying): %s", e)
                time.sleep(2)
            else:
                logger.error("Classifier failed after %d attempts: %s", retries, e)
                return "non_rag"
    return "non_rag"


def calculate_jaccard_similarity(text1: str, text2: str) -> tuple[bool, float]:
    """Jaccard similarity; returns (is_match >= 0.6, score)."""
    words1 = set(re.findall(r"\b\w+\b", text1))
    words2 = set(re.findall(r"\b\w+\b", text2))
    if not words1 or not words2:
        return False, 0.0
    inter = len(words1 & words2)
    union = len(words1 | words2)
    score = inter / union if union else 0.0
    return score >= 0.6, score


def is_semantic_match(
    question1: str,
    question2: str,
    client: DocProcessingLLMClient | None,
    model: str,
) -> tuple[bool, float]:
    """Use LLM to decide if two questions are semantically equivalent. Fallback to Jaccard if no client or model."""
    if not question1 or not question2:
        return False, 0.0
    q1, q2 = question1.lower().strip(), question2.lower().strip()
    if q1 == q2:
        return True, 1.0
    if not client or not model:
        return calculate_jaccard_similarity(q1, q2)
    prompt = f"""Determine if these two questions are asking for the same information, even if phrased differently:

Question 1: {question1}
Question 2: {question2}

Reply with ONLY one word: either "yes" (they are semantically equivalent) or "no" (they are different questions)."""
    for attempt in range(2):
        try:
            response = client.complete(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that determines if questions are asking for the same information."},
                    {"role": "user", "content": prompt},
                ],
            )
            result = str(response.get("content", "")).strip().lower()
            if "yes" in result:
                return True, 0.9
            if "no" in result:
                return False, 0.1
        except Exception as e:
            logger.warning("Semantic match API error: %s; using Jaccard", e)
    return calculate_jaccard_similarity(q1, q2)


def find_matching_qa_id(
    question: str,
    classifier_client: DocProcessingLLMClient | None,
    model: str,
) -> tuple[int | None, float]:
    """Find best matching frequent_qa_pairs id for the question. Uses session from get_session; call within session context or pass session."""
    if not question:
        return None, 0.0
    with get_session() as session:
        pairs = session.query(FrequentQAPair.id, FrequentQAPair.question).filter(
            FrequentQAPair.is_active.is_(True)
        ).all()
    if not pairs:
        return None, 0.0
    norm_q = question.lower().strip()
    candidates: list[tuple[int, str, float]] = []
    for qa_id, qa_question in pairs:
        qa_norm = qa_question.lower().strip()
        if norm_q == qa_norm:
            return qa_id, 1.0
        is_match, score = calculate_jaccard_similarity(norm_q, qa_norm)
        if score > 0.3:
            candidates.append((qa_id, qa_question, score))
    candidates.sort(key=lambda x: x[2], reverse=True)
    top = candidates[:5]
    best_id: int | None = None
    best_score = 0.0
    for qa_id, qa_question, _ in top:
        match, conf = is_semantic_match(question, qa_question, classifier_client, model)
        if match and conf > best_score:
            best_id = qa_id
            best_score = conf
            if conf > 0.9:
                break
    if best_score >= 0.7:
        return best_id, best_score
    return None, best_score


def process_feedback_records(
    config: dict,
    last_processed_id_path: str | Path,
    categories_path: str | Path | None = None,
    classifier_model: str | None = None,
) -> None:
    """
    Process new feedback rows: classify, match to frequent QA, insert into feedback_question_aliases.
    Uses Postgres (get_session) and config for classifier client and categories file.
    """
    last_id = get_last_processed_id(last_processed_id_path)
    categories, category_descriptions = load_categories(categories_path)
    client = _get_classifier_client(config)
    model = classifier_model or config.get("feedback_classifier_model")
    if not model and client:
        logger.warning("feedback_classifier_model not set in config; classification skipped (category=non_rag)")
    model = model or ""

    with get_session() as session:
        rows = (
            session.query(Feedback)
            .filter(Feedback.id > last_id)
            .order_by(Feedback.id)
            .all()
        )
    if not rows:
        logger.debug("No new feedback records to process")
        return

    logger.info("Processing %d new feedback records", len(rows))
    current_max_id = last_id

    for row in rows:
        if row.id > current_max_id:
            current_max_id = row.id
        try:
            category = classify_question(
                row.question or "",
                row.is_rag,
                categories,
                category_descriptions,
                client,
                model=model,
                retries=3,
            )
            qa_id: int | None = None
            match_confidence = 0.0
            if row.is_rag == 1:
                qa_id, match_confidence = find_matching_qa_id(row.question or "", client, model)
            is_match = qa_id is not None
            alias_text = (row.question or "") if is_match else None
            question_rewritten = row.question or ""

            with get_session() as session:
                session.add(
                    FeedbackQuestionAlias(
                        qa_id=qa_id,
                        alias_text=alias_text,
                        session_id=row.session_id,
                        response_id=row.response_id,
                        rating=row.rating,
                        question=row.question or "",
                        question_rewritten=question_rewritten,
                        answer=row.response or "",
                        category=category,
                        is_match=is_match,
                        match_confidence=match_confidence,
                        created_at=row.created_at,
                        notes=f"Processed from feedback. is_rag={row.is_rag}",
                    )
                )
            logger.debug("Processed feedback id=%s category=%s match_confidence=%.2f", row.id, category, match_confidence)
        except Exception as e:
            logger.error("Error processing feedback id=%s: %s", row.id, e, exc_info=True)

    save_last_processed_id(last_processed_id_path, current_max_id)
    logger.info("Feedback processing done. Last processed id=%s", current_max_id)
