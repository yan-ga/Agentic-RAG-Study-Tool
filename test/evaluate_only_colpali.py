"""
Evaluation script for ColPali-only retrieval.
Tests on easy.jsonl and hard.jsonl test files.
Uses the retrieval strategy from ask_only_colpali.py.
"""

import os
import json
import numpy as np
import torch
from transformers import ColPaliForRetrieval, ColPaliProcessor
from collections import defaultdict

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
CHUNKS_PATH = os.path.join(DATA_DIR, "new_chunks.jsonl")
COLPALI_EMBEDS_PATH = os.path.join(DATA_DIR, "colpali_page_emb.pt")
EASY_PATH = os.path.join(SCRIPT_DIR, "easy.jsonl")
HARD_PATH = os.path.join(SCRIPT_DIR, "hard.jsonl")

COLPALI_NAME = "vidore/colpali-v1.3-hf"


def load_chunks(chunks_jsonl):
    """Load chunks from JSONL file."""
    chunks = []
    with open(chunks_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def load_test_data(test_jsonl):
    """Load test questions from JSONL file."""
    tests = []
    with open(test_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tests.append(json.loads(line))
    return tests


def retrieve_colpali_only(chunks, chunk_ids, colpali_model, colpali_processor,
                          colpali_page_embeds, query, top_k=10):
    """
    ColPali-only retrieval.
    Returns list of (chunk_id, score) tuples.
    """
    colpali_model.eval()
    with torch.no_grad():
        q_inputs = colpali_processor(text=[query], return_tensors="pt").to(colpali_model.device)
        q_emb = colpali_model(**q_inputs).embeddings
        colpali_scores = colpali_processor.score_retrieval(q_emb, colpali_page_embeds)
        colpali_scores = colpali_scores[0].cpu().numpy().astype(np.float32)

    order = np.argsort(-colpali_scores)[:top_k]
    return [(chunk_ids[i], colpali_scores[i]) for i in order]


def evaluate_retrieval(test_data, chunks, chunk_ids, colpali_model, colpali_processor,
                       colpali_page_embeds, top_k_values=[1, 3, 5, 10]):
    """
    Evaluate retrieval performance.

    Metrics:
    - Recall@k: fraction of relevant items found in top-k
    - Hit@k: fraction of queries with at least one relevant item in top-k
    - MRR: Mean Reciprocal Rank
    """
    metrics = defaultdict(list)

    for test in test_data:
        question = test["question"]
        relevant_ids = set(test["relevant_chunk_ids"])

        # Get top-k results (use max k)
        max_k = max(top_k_values)
        results = retrieve_colpali_only(
            chunks, chunk_ids, colpali_model, colpali_processor,
            colpali_page_embeds, question, top_k=max_k
        )
        retrieved_ids = [r[0] for r in results]

        # Compute MRR (find first relevant result)
        rr = 0.0
        for rank, chunk_id in enumerate(retrieved_ids, 1):
            if chunk_id in relevant_ids:
                rr = 1.0 / rank
                break
        metrics["mrr"].append(rr)

        # Compute Recall@k and Hit@k for each k
        for k in top_k_values:
            top_k_ids = set(retrieved_ids[:k])
            hits = len(top_k_ids & relevant_ids)
            recall = hits / len(relevant_ids) if relevant_ids else 0.0
            hit = 1.0 if hits > 0 else 0.0

            metrics[f"recall@{k}"].append(recall)
            metrics[f"hit@{k}"].append(hit)

    # Average metrics
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    return avg_metrics


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print("=" * 60)
    print("ColPali-Only Retrieval Evaluation")
    print("=" * 60)

    # Load data
    print("\nLoading chunks...")
    chunks = load_chunks(CHUNKS_PATH)
    chunk_ids = [c["chunk_id"] for c in chunks]
    print(f"  Loaded {len(chunks)} chunks")

    print("Loading ColPali page embeddings...")
    colpali_page_embeds = torch.load(COLPALI_EMBEDS_PATH, weights_only=True)
    print(f"  ColPali embeddings: {len(colpali_page_embeds)} pages")

    print("Loading ColPali model...")
    colpali_model = ColPaliForRetrieval.from_pretrained(
        COLPALI_NAME, torch_dtype=torch.float32, device_map="cpu"
    )
    colpali_processor = ColPaliProcessor.from_pretrained(COLPALI_NAME)

    # Load test data
    print("\nLoading test data...")
    easy_tests = load_test_data(EASY_PATH)
    hard_tests = load_test_data(HARD_PATH)
    print(f"  Easy: {len(easy_tests)} questions")
    print(f"  Hard: {len(hard_tests)} questions")

    top_k_values = [1, 3, 5, 10]

    print("\nUsing ColPali-only retrieval")
    print("=" * 60)

    # Evaluate on easy
    print("\n[EASY TEST SET]")
    easy_metrics = evaluate_retrieval(
        easy_tests, chunks, chunk_ids, colpali_model, colpali_processor,
        colpali_page_embeds, top_k_values
    )
    print(f"  MRR:       {easy_metrics['mrr']:.4f}")
    for k in top_k_values:
        print(f"  Hit@{k}:    {easy_metrics[f'hit@{k}']:.4f}  |  Recall@{k}: {easy_metrics[f'recall@{k}']:.4f}")

    # Evaluate on hard
    print("\n[HARD TEST SET]")
    hard_metrics = evaluate_retrieval(
        hard_tests, chunks, chunk_ids, colpali_model, colpali_processor,
        colpali_page_embeds, top_k_values
    )
    print(f"  MRR:       {hard_metrics['mrr']:.4f}")
    for k in top_k_values:
        print(f"  Hit@{k}:    {hard_metrics[f'hit@{k}']:.4f}  |  Recall@{k}: {hard_metrics[f'recall@{k}']:.4f}")


if __name__ == "__main__":
    main()
