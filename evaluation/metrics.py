"""
metrics.py
----------
Standalone metric calculations for RAG evaluation.
These are used by evaluator.py and the dashboard.

Metrics:
  - Retrieval Precision@K  : of K chunks retrieved, how many relevant?
  - Answer F1              : token overlap between answer and ground truth
  - Exact Match            : does answer contain ground truth substring?
  - Context Utilization    : how much of retrieved context was used?
  - Hallucination Rate     : % of eval runs with hallucination detected
"""

import re
import string
from collections import Counter
from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    precision_at_k: float      # relevant_retrieved / k
    recall_at_k: float         # relevant_retrieved / total_relevant
    f1_at_k: float


@dataclass
class AnswerMetrics:
    f1_score: float
    exact_match: bool
    token_overlap: float


def normalize_text(text: str) -> str:
    """Lowercase, remove punctuation and extra whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def compute_answer_f1(prediction: str, ground_truth: str) -> float:
    """
    Token-level F1 between predicted answer and ground truth.
    Standard metric from SQuAD evaluation.
    """
    pred_tokens   = normalize_text(prediction).split()
    truth_tokens  = normalize_text(ground_truth).split()

    if not pred_tokens or not truth_tokens:
        return 0.0

    pred_counter  = Counter(pred_tokens)
    truth_counter = Counter(truth_tokens)

    common = sum((pred_counter & truth_counter).values())

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall    = common / len(truth_tokens)
    f1        = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def compute_exact_match(prediction: str, ground_truth: str) -> bool:
    """Check if ground truth appears in the prediction (case-insensitive)."""
    return normalize_text(ground_truth) in normalize_text(prediction)


def compute_token_overlap(prediction: str, ground_truth: str) -> float:
    """What fraction of ground truth tokens appear in prediction."""
    pred_tokens  = set(normalize_text(prediction).split())
    truth_tokens = set(normalize_text(ground_truth).split())

    if not truth_tokens:
        return 0.0

    overlap = len(pred_tokens & truth_tokens)
    return round(overlap / len(truth_tokens), 4)


def compute_answer_metrics(prediction: str, ground_truth: str) -> AnswerMetrics:
    return AnswerMetrics(
        f1_score=compute_answer_f1(prediction, ground_truth),
        exact_match=compute_exact_match(prediction, ground_truth),
        token_overlap=compute_token_overlap(prediction, ground_truth),
    )


def compute_retrieval_precision(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
    k: int | None = None,
) -> RetrievalMetrics:
    """
    Compute Precision@K, Recall@K, F1@K.

    Args:
        retrieved_chunk_ids: chunk IDs returned by retrieval pipeline
        relevant_chunk_ids:  ground truth relevant chunk IDs
        k:                   cutoff (default: len(retrieved))
    """
    k = k or len(retrieved_chunk_ids)
    retrieved_at_k = set(retrieved_chunk_ids[:k])
    relevant       = set(relevant_chunk_ids)

    if not retrieved_at_k:
        return RetrievalMetrics(0.0, 0.0, 0.0)

    tp = len(retrieved_at_k & relevant)

    precision = tp / len(retrieved_at_k)
    recall    = tp / len(relevant) if relevant else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return RetrievalMetrics(
        precision_at_k=round(precision, 4),
        recall_at_k=round(recall, 4),
        f1_at_k=round(f1, 4),
    )


def compute_context_utilization(
    answer: str,
    context_chunks: list[dict],
) -> float:
    """
    What fraction of retrieved context chunks contributed to the answer?
    Measured by checking if chunk content words appear in the answer.
    """
    if not context_chunks:
        return 0.0

    answer_words = set(normalize_text(answer).split())
    utilized = 0

    for chunk in context_chunks:
        chunk_words = set(normalize_text(chunk.get("content", "")).split())
        # Consider chunk "utilized" if >10% of its words appear in answer
        if chunk_words:
            overlap = len(answer_words & chunk_words) / len(chunk_words)
            if overlap > 0.10:
                utilized += 1

    return round(utilized / len(context_chunks), 4)


def compute_hallucination_rate(eval_runs: list[dict]) -> float:
    """
    Compute hallucination rate across a batch of eval runs.

    Args:
        eval_runs: list of dicts with key 'hallucination_detected' (bool)

    Returns:
        fraction of runs where hallucination was detected
    """
    if not eval_runs:
        return 0.0

    detected = sum(1 for r in eval_runs if r.get("hallucination_detected", False))
    return round(detected / len(eval_runs), 4)


def aggregate_eval_scores(eval_runs: list[dict]) -> dict:
    """
    Aggregate metrics across multiple eval runs.
    Input: list of eval score dicts from evaluator.py

    Returns:
        dict with mean/min/max for each metric
    """
    if not eval_runs:
        return {}

    metrics = ["groundedness", "relevance", "coherence", "overall"]
    result  = {}

    for metric in metrics:
        values = [r[metric] for r in eval_runs if metric in r]
        if values:
            result[f"{metric}_mean"] = round(sum(values) / len(values), 4)
            result[f"{metric}_min"]  = round(min(values), 4)
            result[f"{metric}_max"]  = round(max(values), 4)

    result["hallucination_rate"]  = compute_hallucination_rate(eval_runs)
    result["total_runs"]          = len(eval_runs)

    return result