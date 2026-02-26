"""
synthetic_dataset.py
--------------------
Automatically generates evaluation Q&A pairs from indexed documents.

Why synthetic data?
  Most teams skip evaluation because building a test dataset
  manually is tedious. This script auto-generates 100 diverse
  questions with ground truth answers directly from your documents.

Question types generated:
  1. Factual       — "What is X?" (simple lookup)
  2. Reasoning     — "Why did X happen?" (multi-sentence answer)
  3. Multi-hop     — "How does X relate to Y?" (cross-chunk)
  4. Tabular       — "What was the value in column X row Y?"
  5. Adversarial   — questions the document CANNOT answer (tests robustness)

Output saved to: data/eval_dataset/eval_questions.json
"""

import json
import uuid
from pathlib import Path
from loguru import logger
from openai import AzureOpenAI

from config import get_settings


QUESTION_GEN_PROMPT = """You are building an evaluation dataset for a RAG (Retrieval Augmented Generation) system.

Given the following document chunk, generate {n_questions} diverse questions that:
1. Can be answered using ONLY this chunk
2. Cover different difficulty levels and question types
3. Have clear, verifiable answers

Question types to include (mix them):
- FACTUAL: Simple "what/who/when/where" questions
- REASONING: "Why/how" questions requiring explanation
- TABULAR: Questions about numbers or table values (if tables present)
- ADVERSARIAL: A question that CANNOT be answered from this chunk (answer = "not found")

Document chunk:
{chunk}

Source: {doc_name}, Page {page_number}

Return ONLY a JSON array. Each item must have:
- "question": the question string
- "ground_truth": the expected answer (verbatim from chunk or "Information not found in document")
- "question_type": one of [factual, reasoning, tabular, adversarial]
- "doc_name": "{doc_name}"
- "page_number": {page_number}
- "chunk_id": "{chunk_id}"

JSON array only, no markdown, no explanation:"""


def generate_questions_from_chunk(
    chunk_content: str,
    doc_name: str,
    page_number: int,
    chunk_id: str,
    n_questions: int = 3,
    client: AzureOpenAI = None,
    model: str = None,
) -> list[dict]:
    """Generate Q&A pairs from a single chunk."""
    settings = get_settings()

    if client is None:
        client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_key,
            api_version=settings.azure_openai_api_version,
        )

    if model is None:
        model = settings.azure_openai_deployment

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": QUESTION_GEN_PROMPT.format(
                    n_questions=n_questions,
                    chunk=chunk_content[:1500],
                    doc_name=doc_name,
                    page_number=page_number,
                    chunk_id=chunk_id,
                ),
            }],
            temperature=0.8,
            max_tokens=800,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        questions = json.loads(raw.strip())

        # Add unique IDs
        for q in questions:
            q["id"] = str(uuid.uuid4())

        return questions

    except Exception as e:
        logger.warning(f"Question generation failed for chunk {chunk_id}: {e}")
        return []


def generate_eval_dataset(
    n_chunks: int = 30,
    questions_per_chunk: int = 3,
    output_path: str = "data/eval_dataset/eval_questions.json",
) -> list[dict]:
    """
    Generate a full evaluation dataset from indexed documents.

    Args:
        n_chunks:             how many chunks to sample from the index
        questions_per_chunk:  questions to generate per chunk
        output_path:          where to save the dataset

    Returns:
        list of Q&A dicts
    """
    settings = get_settings()

    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient

    search_client = SearchClient(
        endpoint=settings.azure_search_endpoint,
        index_name=settings.azure_search_index,
        credential=AzureKeyCredential(settings.azure_search_key),
    )

    # Sample diverse chunks from the index
    logger.info(f"Sampling {n_chunks} chunks from index for eval dataset generation")

    results = list(search_client.search(
        search_text="*",
        select=["id", "doc_id", "doc_name", "page_number", "chunk_type", "content"],
        top=n_chunks,
    ))

    if not results:
        logger.error("No chunks found in index — run ingestion first")
        return []

    client = AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_key,
        api_version=settings.azure_openai_api_version,
    )

    all_questions = []

    for i, chunk in enumerate(results):
        logger.info(f"Generating questions for chunk {i+1}/{len(results)}: {chunk['id']}")

        questions = generate_questions_from_chunk(
            chunk_content=chunk["content"],
            doc_name=chunk["doc_name"],
            page_number=chunk["page_number"],
            chunk_id=chunk["id"],
            n_questions=questions_per_chunk,
            client=client,
            model=settings.azure_openai_deployment,
        )

        all_questions.extend(questions)

        # Rate limit protection
        import time
        time.sleep(0.5)

    # Save dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_questions, f, indent=2)

    logger.info(
        f"Eval dataset saved: {len(all_questions)} questions → {output_path}"
    )

    # Print summary
    types = {}
    for q in all_questions:
        t = q.get("question_type", "unknown")
        types[t] = types.get(t, 0) + 1

    logger.info(f"Question type breakdown: {types}")

    return all_questions


def load_eval_dataset(
    path: str = "data/eval_dataset/eval_questions.json",
) -> list[dict]:
    """Load an existing eval dataset from disk."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Eval dataset not found at {path}. "
            "Run generate_eval_dataset() first."
        )
    with open(p) as f:
        return json.load(f)


if __name__ == "__main__":
    dataset = generate_eval_dataset(
        n_chunks=20,
        questions_per_chunk=3,
    )
    print(f"\nGenerated {len(dataset)} evaluation questions")
    print("\nSample questions:")
    for q in dataset[:3]:
        print(f"\n  [{q['question_type'].upper()}] {q['question']}")
        print(f"  Answer: {q['ground_truth'][:100]}...")