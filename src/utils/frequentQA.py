"""
Frequent QA similarity (PostgreSQL). Uses database.get_session() and models.FrequentQAPair / QATable.
"""
import math
import re
from collections import Counter
from difflib import SequenceMatcher

from database import get_session
from models import FrequentQAPair, QATable


def normalize_question(question):
    q = question.strip()
    q = re.sub(r'[，。、？！：；""''（）【】［］｛｝《》〈〉「」『』〔〕…—－～]', ' ', q)
    q = re.sub(r'(?i)lotus\s+technology(?:\'s)?', '', question)
    return q


def calculate_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def compare_questions(question1, question2, use_normalization=True):
    if use_normalization:
        q1_normalized = normalize_question(question1)
        q2_normalized = normalize_question(question2)
        similarity = calculate_similarity(q1_normalized, q2_normalized)
    else:
        q1_normalized = question1
        q2_normalized = question2
        similarity = calculate_similarity(question1, question2)
    return similarity, [q1_normalized, q2_normalized]


def periods_to_dict(row_id: int) -> dict:
    """Return period_data for qa_table row (PostgreSQL: single JSONB column)."""
    with get_session() as session:
        row = session.query(QATable).filter(QATable.id == row_id).first()
        if row is None or row.period_data is None:
            return {}
        return dict(row.period_data)


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = len(corpus)
        self.avg_doc_len = sum(len(doc) for doc in corpus) / self.corpus_size
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.initialize(corpus)

    def initialize(self, corpus):
        for document in corpus:
            self.doc_len.append(len(document))
            freq = Counter(document)
            self.doc_freqs.append(freq)
            for word, count in freq.items():
                if word not in self.idf:
                    self.idf[word] = 0
                self.idf[word] += 1
        for word, doc_freq in self.idf.items():
            self.idf[word] = math.log(
                (self.corpus_size - doc_freq + 0.5) / (doc_freq + 0.5) + self.epsilon
            )

    def score(self, query, index):
        score = 0.0
        doc_len = self.doc_len[index]
        frequencies = self.doc_freqs[index]
        for word in query:
            if word not in frequencies:
                continue
            freq = frequencies[word]
            numerator = self.idf[word] * freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            score += numerator / denominator
        return score

    def get_scores(self, query):
        return [self.score(query, i) for i in range(self.corpus_size)]


class QuestionSimilarityFinder:
    """Uses PostgreSQL (database_url must be set; init_db called by app)."""

    def __init__(self, database_url: str | None = None, **_kwargs):
        if database_url:
            from database import init_db
            init_db(database_url)
        self._database_url = database_url

    def find_similar_questions_db(self, input_question, top_n=5, threshold=0.55, use_normalization=True):
        with get_session() as session:
            rows = session.query(FrequentQAPair).filter(FrequentQAPair.is_active.is_(True)).all()
        results = []
        for row in rows:
            similarity, q_normalized = compare_questions(input_question, row.question_rewritten, use_normalization)
            if similarity >= threshold:
                results.append((row.id, row.question, row.question_rewritten, row.answer, similarity, q_normalized))
        results.sort(key=lambda x: x[4], reverse=True)
        return results[:top_n]

    def find_similar_questions_table(self, input_question, top_n=5, threshold=0.55, use_normalization=True):
        with get_session() as session:
            rows = session.query(QATable).filter(QATable.is_active.is_(True)).all()
        results = []
        for row in rows:
            similarity, q_normalized = compare_questions(input_question, row.question_rewritten, use_normalization)
            if similarity >= threshold:
                results.append((
                    row.id,
                    row.question,
                    row.question_rewritten,
                    periods_to_dict(row.id),
                    similarity,
                    q_normalized,
                ))
        results.sort(key=lambda x: x[4], reverse=True)
        return results[:top_n]

    def find_similar_questions_bm25_db(self, input_question, top_n=5, threshold=3.0):
        normalized_input = normalize_question(input_question)
        tokenized_input = normalized_input.split()
        with get_session() as session:
            rows = session.query(FrequentQAPair).filter(FrequentQAPair.is_active.is_(True)).all()
        corpus = []
        db_questions = []
        for row in rows:
            normalized_question = normalize_question(row.question_rewritten)
            tokenized_question = normalized_question.split()
            corpus.append(tokenized_question)
            db_questions.append((row.id, row.question, row.question_rewritten, row.answer, normalized_question))
        bm25 = BM25(corpus)
        scores = bm25.get_scores(tokenized_input)
        results = []
        for i, score in enumerate(scores):
            if score >= threshold:
                row_id, question, question_rewritten, answer, normalized_question = db_questions[i]
                results.append((row_id, question, question_rewritten, answer, score, [normalized_input, normalized_question]))
        results.sort(key=lambda x: x[4], reverse=True)
        return results[:top_n]

    def find_similar_questions_bm25_table(self, input_question, top_n=5, threshold=3.0):
        normalized_input = normalize_question(input_question)
        tokenized_input = normalized_input.split()
        with get_session() as session:
            rows = session.query(QATable).filter(QATable.is_active.is_(True)).all()
        corpus = []
        db_questions = []
        for row in rows:
            normalized_question = normalize_question(row.question_rewritten)
            tokenized_question = normalized_question.split()
            corpus.append(tokenized_question)
            db_questions.append((row.id, row.question, row.question_rewritten, normalized_question))
        bm25 = BM25(corpus)
        scores = bm25.get_scores(tokenized_input)
        results = []
        for i, score in enumerate(scores):
            if score >= threshold:
                row_id, question, question_rewritten, normalized_question = db_questions[i]
                results.append((
                    row_id,
                    question,
                    question_rewritten,
                    periods_to_dict(row_id),
                    score,
                    [normalized_input, normalized_question],
                ))
        results.sort(key=lambda x: x[4], reverse=True)
        return results[:top_n]

    def get_full_qa_by_id(self, question_id):
        with get_session() as session:
            row = session.query(FrequentQAPair).filter(FrequentQAPair.id == question_id).first()
        if row is None:
            return None
        return {
            "id": row.id,
            "question": row.question,
            "question_rewritten": row.question_rewritten,
            "answer": row.answer,
            "category": row.category,
            "last_updated": row.last_updated,
            "updated_by": row.updated_by,
            "is_active": row.is_active,
            "view_count": row.view_count,
            "version": row.version,
            "tags": row.tags,
            "metadata": row.metadata_,
        }


if __name__ == "__main__":
    from config import get_config
    config = get_config()
    finder = QuestionSimilarityFinder(database_url=config.get("database_url"))
    # Example usage
    results = finder.find_similar_questions_db("What is the gross margin of Lotus Technology?", top_n=3)
    print(results)
