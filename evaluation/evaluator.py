"""
evaluator.py
------------
Scores RAG responses using Azure AI Evaluation:
  - Groundedness : is every claim supported by context?
  - Relevance    : does answer address the question?
  - Coherence    : is the answer well-structured?
  - Similarity   : semantic match vs ground truth (if available)

Also logs all scores to MLflow for tracking over time.
"""

import json
import time
from dataclasses import dataclass
from loguru import logger

import mlflow
from openai import AzureOpenAI

from config import get_settings


@dataclass
class EvalScores:
    groundedness: float   # 1-5
    relevance: float      # 1-5
    coherence: float      # 1-5
    similarity: float     # 0-1 (only if ground_truth provided)
    hallucination_detected: bool
    overall: float        # weighted average


GROUNDEDNESS_PROMPT = """You are evaluating whether an AI answer is grounded in the provided context.

Score the answer from 1-5:
5 = Every claim is directly supported by the context
4 = Most claims supported, minor extrapolation
3 = Some claims supported, some not verifiable
2 = Many claims not in context
1 = Answer contradicts or ignores context entirely

Context:
{context}

Answer:
{answer}

Return ONLY a JSON object: {{"score": X, "reasoning": "brief explanation", "hallucination": true/false}}"""


RELEVANCE_PROMPT = """Score how well this answer addresses the question from 1-5:
5 = Directly and completely answers the question
4 = Mostly answers with minor gaps
3 = Partially answers
2 = Tangentially related
1 = Does not answer the question

Question: {query}
Answer: {answer}

Return ONLY a JSON object: {{"score": X, "reasoning": "brief explanation"}}"""


COHERENCE_PROMPT = """Score the coherence and quality of this answer from 1-5:
5 = Well-structured, clear, logical flow
4 = Mostly clear with minor issues
3 = Somewhat clear but disorganized
2 = Hard to follow
1 = Incoherent

Answer: {answer}

Return ONLY a JSON object: {{"score": X, "reasoning": "brief explanation"}}"""


def _gpt_score(prompt: str, client: AzureOpenAI, model: str) -> dict:
    """Call GPT to get an evaluation score."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Eval scoring failed: {e}")
        return {"score": 3.0, "reasoning": f"eval error: {e}"}


def evaluate_answer(
    query: str,
    answer: str,
    context_chunks: list[dict],
    ground_truth: str | None = None,
    log_to_mlflow: bool = True,
) -> EvalScores:
    """
    Score a RAG response across all dimensions.

    Args:
        query:          user question
        answer:         generated answer
        context_chunks: retrieved context used
        ground_truth:   expected answer (optional)
        log_to_mlflow:  whether to log scores to MLflow

    Returns:
        EvalScores with all dimensions scored
    """
    settings = get_settings()
    client   = AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_key,
        api_version=settings.azure_openai_api_version,
    )

    context_text = "\n\n".join(c["content"] for c in context_chunks)

    # Score all three dimensions
    g_result = _gpt_score(
        GROUNDEDNESS_PROMPT.format(context=context_text[:3000], answer=answer),
        client, settings.azure_openai_deployment
    )
    r_result = _gpt_score(
        RELEVANCE_PROMPT.format(query=query, answer=answer),
        client, settings.azure_openai_deployment
    )
    c_result = _gpt_score(
        COHERENCE_PROMPT.format(answer=answer),
        client, settings.azure_openai_deployment
    )

    groundedness = float(g_result.get("score", 3))
    relevance    = float(r_result.get("score", 3))
    coherence    = float(c_result.get("score", 3))
    hallucination = bool(g_result.get("hallucination", False))

    # Similarity vs ground truth
    similarity = 0.0
    if ground_truth:
        similarity = _compute_similarity(answer, ground_truth, client, settings.azure_openai_deployment)

    # Weighted overall score (groundedness weighted highest)
    overall = (groundedness * 0.4 + relevance * 0.35 + coherence * 0.25) / 5.0

    scores = EvalScores(
        groundedness=groundedness,
        relevance=relevance,
        coherence=coherence,
        similarity=similarity,
        hallucination_detected=hallucination,
        overall=overall,
    )

    logger.info(
        f"Eval scores — "
        f"groundedness:{groundedness:.1f} "
        f"relevance:{relevance:.1f} "
        f"coherence:{coherence:.1f} "
        f"hallucination:{hallucination} "
        f"overall:{overall:.3f}"
    )

    if log_to_mlflow:
        _log_to_mlflow(query, answer, scores)

    return scores


def _compute_similarity(answer: str, ground_truth: str, client: AzureOpenAI, model: str) -> float:
    """Compute semantic similarity between answer and ground truth."""
    try:
        response = client.embeddings.create(
            input=[answer, ground_truth],
            model=get_settings().azure_openai_emb_deployment,
        )
        import numpy as np
        a = np.array(response.data[0].embedding)
        b = np.array(response.data[1].embedding)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    except Exception:
        return 0.0


def _log_to_mlflow(query: str, answer: str, scores: EvalScores) -> None:
    """Log evaluation scores to MLflow."""
    try:
        with mlflow.start_run(run_name="rag-eval", nested=True):
            mlflow.log_metrics({
                "groundedness":         scores.groundedness,
                "relevance":            scores.relevance,
                "coherence":            scores.coherence,
                "similarity":           scores.similarity,
                "overall":              scores.overall,
                "hallucination_detected": int(scores.hallucination_detected),
            })
            mlflow.log_params({
                "query_length": len(query),
                "answer_length": len(answer),
            })
    except Exception as e:
        logger.warning(f"MLflow logging failed (non-critical): {e}")