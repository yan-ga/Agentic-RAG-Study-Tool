"""
Evaluation script for Qwen3-VL-Embedding-only retrieval.
Tests on easy.jsonl and hard.jsonl test files.
Uses Qwen3-VL-Embedding for both query and page embeddings.
"""

import os
import sys
import json
import numpy as np
import torch
from collections import defaultdict

# Add the Qwen3-VL-Embedding source to path
sys.path.insert(0, "/Users/yan_ga/Desktop/Replicate Qwen-VL-Embedding/Qwen3-VL-Embedding-Mac/src")
from models.qwen3_vl_embedding import Qwen3VLEmbedder

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
CHUNKS_PATH = os.path.join(DATA_DIR, "new_chunks.jsonl")
QWEN3VL_EMBEDS_PATH = os.path.join(DATA_DIR, "qwen3vl_page_emb.npy")
EASY_PATH = os.path.join(SCRIPT_DIR, "easy.jsonl")
HARD_PATH = os.path.join(SCRIPT_DIR, "hard.jsonl")

MODEL_PATH = "/Users/yan_ga/Desktop/Replicate Qwen-VL-Embedding/Qwen3-VL-Embedding-Mac/models/Qwen3-VL-Embedding-2B"

# Instructions for retrieval task
QUERY_INSTRUCTION = "Represent this query for retrieving relevant lecture slides."


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


def retrieve_qwen3vl(chunk_ids, embedder, page_embeds, query, top_k=10):
    """
    Qwen3-VL-only retrieval.
    Returns list of (chunk_id, score) tuples.
    """
    # Encode query text with instruction
    q_emb = embedder.process([{
        "text": query,
        "instruction": QUERY_INSTRUCTION
    }])
    q_emb = q_emb.cpu().numpy().astype(np.float32)

    # Compute similarity scores (dot product since both are normalized)
    scores = (page_embeds @ q_emb.T).squeeze()

    order = np.argsort(-scores)[:top_k]
    return [(chunk_ids[i], scores[i]) for i in order]


def evaluate_retrieval(test_data, chunk_ids, embedder, page_embeds,
                       top_k_values=[1, 3, 5, 10]):
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
        results = retrieve_qwen3vl(chunk_ids, embedder, page_embeds, question, top_k=max_k)
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
    print("Qwen3-VL-Embedding Retrieval Evaluation")
    print("=" * 60)

    # Load data
    print("\nLoading chunks...")
    chunks = load_chunks(CHUNKS_PATH)
    chunk_ids = [c["chunk_id"] for c in chunks]
    print(f"  Loaded {len(chunks)} chunks")

    print("Loading Qwen3-VL page embeddings...")
    page_embeds = np.load(QWEN3VL_EMBEDS_PATH)
    print(f"  Page embeddings shape: {page_embeds.shape}")

    print("Loading Qwen3-VL-Embedding model (for query encoding)...")
    embedder = Qwen3VLEmbedder(
        model_name_or_path=MODEL_PATH,
        torch_dtype=torch.float32,  # MPS works better with float32
    )
    print(f"  Model loaded on device: {embedder.model.device}")

    # Load test data
    print("\nLoading test data...")
    easy_tests = load_test_data(EASY_PATH)
    hard_tests = load_test_data(HARD_PATH)
    print(f"  Easy: {len(easy_tests)} questions")
    print(f"  Hard: {len(hard_tests)} questions")

    top_k_values = [1, 3, 5, 10]

    print("\nUsing Qwen3-VL-Embedding-only retrieval")
    print("=" * 60)

    # Evaluate on easy
    print("\n[EASY TEST SET]")
    easy_metrics = evaluate_retrieval(
        easy_tests, chunk_ids, embedder, page_embeds, top_k_values
    )
    print(f"  MRR:       {easy_metrics['mrr']:.4f}")
    for k in top_k_values:
        print(f"  Hit@{k}:    {easy_metrics[f'hit@{k}']:.4f}  |  Recall@{k}: {easy_metrics[f'recall@{k}']:.4f}")

    # Evaluate on hard
    print("\n[HARD TEST SET]")
    hard_metrics = evaluate_retrieval(
        hard_tests, chunk_ids, embedder, page_embeds, top_k_values
    )
    print(f"  MRR:       {hard_metrics['mrr']:.4f}")
    for k in top_k_values:
        print(f"  Hit@{k}:    {hard_metrics[f'hit@{k}']:.4f}  |  Recall@{k}: {hard_metrics[f'recall@{k}']:.4f}")


if __name__ == "__main__":
    main()
