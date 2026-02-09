"""
Evaluation script for multi-modal retrieval with ColPali.
Tests on easy.jsonl and hard.jsonl test files.
Uses the retrieval strategy from ask_colpali.py.
"""

import os
import sys
import json
import re
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import ColPaliForRetrieval, ColPaliProcessor
from collections import defaultdict

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
CHUNKS_PATH = os.path.join(DATA_DIR, "new_chunks.jsonl")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "new_embeddings.npy")
COLPALI_EMBEDS_PATH = os.path.join(DATA_DIR, "colpali_page_emb.pt")
EASY_PATH = os.path.join(SCRIPT_DIR, "easy.jsonl")
HARD_PATH = os.path.join(SCRIPT_DIR, "hard.jsonl")

COLPALI_NAME = "vidore/colpali-v1.3-hf"


def tokenize(text):
    """Tokenize text for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


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


def minmax(x):
    """Min-max normalization."""
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)


def retrieve_hybrid_with_colpali(chunks, chunk_ids, bm25, emb, embed_model,
                                  colpali_model, colpali_processor, colpali_page_embeds,
                                  query, top_k=40, alpha=0.5, gamma=0.5):
    """
    First stage: Hybrid retrieval with BM25 + text embeddings + ColPali.
    Returns candidates, colpali_scores, and candidate indices.
    """
    # BM25 scores
    bm25_scores = np.array(bm25.get_scores(tokenize(query)), dtype=np.float32)

    # Text embedding scores
    q_emb = embed_model.encode([query], normalize_embeddings=True)[0]
    emb_scores = (emb @ q_emb).astype(np.float32)

    # ColPali visual scores
    colpali_model.eval()
    with torch.no_grad():
        q_inputs = colpali_processor(text=[query], return_tensors="pt").to(colpali_model.device)
        q_emb_colpali = colpali_model(**q_inputs).embeddings
        colpali_scores = colpali_processor.score_retrieval(q_emb_colpali, colpali_page_embeds)
        colpali_scores = colpali_scores[0].cpu().numpy().astype(np.float32)

    # Fuse scores
    text_hybrid = alpha * minmax(bm25_scores) + (1 - alpha) * minmax(emb_scores)
    final_scores = (1 - gamma) * text_hybrid + gamma * minmax(colpali_scores)

    order = np.argsort(-final_scores)[:top_k]
    candidates = [(chunks[i], chunk_ids[i], final_scores[i]) for i in order]

    return candidates, colpali_scores, order


def rerank_with_colpali(reranker, query, candidates, colpali_scores, cand_indices,
                        top_k_final=10, beta=0.5):
    """
    Second stage: Rerank with cross-encoder + ColPali fusion.
    Returns list of (chunk_id, score) tuples.
    """
    if not candidates:
        return []

    # Cross-encoder reranking
    pairs = [(query, c[0]["text"]) for c in candidates]
    rerank_scores = reranker.predict(pairs, batch_size=16).astype(np.float32)

    # ColPali scores for candidates
    c_colpali_scores = colpali_scores[cand_indices].astype(np.float32)

    # Final fusion
    final_scores = (1 - beta) * minmax(rerank_scores) + beta * minmax(c_colpali_scores)
    order = np.argsort(-final_scores)[:top_k_final]

    reranked = [(candidates[i][1], final_scores[i]) for i in order]
    return reranked


def retrieve_full_pipeline(chunks, chunk_ids, bm25, emb, embed_model,
                           colpali_model, colpali_processor, colpali_page_embeds,
                           reranker, query,
                           top_k_recall=40, top_k_final=10,
                           alpha=0.5, gamma=0.5, beta=0.5):
    """
    Full multi-modal retrieval pipeline with ColPali.
    Returns list of (chunk_id, score) tuples.
    """
    # Stage 1: Hybrid retrieval with ColPali
    candidates, colpali_scores, cand_indices = retrieve_hybrid_with_colpali(
        chunks, chunk_ids, bm25, emb, embed_model,
        colpali_model, colpali_processor, colpali_page_embeds, query,
        top_k=top_k_recall, alpha=alpha, gamma=gamma
    )

    # Stage 2: Rerank with cross-encoder + ColPali fusion
    results = rerank_with_colpali(
        reranker, query, candidates, colpali_scores, cand_indices,
        top_k_final=top_k_final, beta=beta
    )

    return results


def evaluate_retrieval(test_data, chunks, chunk_ids, bm25, emb, embed_model,
                       colpali_model, colpali_processor, colpali_page_embeds,
                       reranker, top_k_values=[1, 3, 5, 10], top_k_recall=40,
                       alpha=0.5, gamma=0.5, beta=0.5):
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
        results = retrieve_full_pipeline(
            chunks, chunk_ids, bm25, emb, embed_model,
            colpali_model, colpali_processor, colpali_page_embeds,
            reranker, question,
            top_k_recall=top_k_recall, top_k_final=max_k,
            alpha=alpha, gamma=gamma, beta=beta
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
    print("Multi-Modal Retrieval with ColPali Evaluation")
    print("=" * 60)

    # Load data
    print("\nLoading chunks...")
    chunks = load_chunks(CHUNKS_PATH)
    chunk_ids = [c["chunk_id"] for c in chunks]
    print(f"  Loaded {len(chunks)} chunks")

    print("Loading text embeddings...")
    emb = np.load(EMBEDDINGS_PATH)
    print(f"  Text embeddings shape: {emb.shape}")

    print("Loading ColPali page embeddings...")
    colpali_page_embeds = torch.load(COLPALI_EMBEDS_PATH, weights_only=True)
    print(f"  ColPali embeddings: {len(colpali_page_embeds)} pages")

    print("Building BM25 index...")
    corpus_tokens = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)

    print("Loading sentence transformer model...")
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Loading ColPali model...")
    colpali_model = ColPaliForRetrieval.from_pretrained(
        COLPALI_NAME, torch_dtype=torch.float32, device_map="cpu"
    )
    colpali_processor = ColPaliProcessor.from_pretrained(COLPALI_NAME)

    print("Loading cross-encoder reranker...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Load test data
    print("\nLoading test data...")
    easy_tests = load_test_data(EASY_PATH)
    hard_tests = load_test_data(HARD_PATH)
    print(f"  Easy: {len(easy_tests)} questions")
    print(f"  Hard: {len(hard_tests)} questions")

    # Parameters (as in ask_colpali.py)
    top_k_values = [1, 3, 5, 10]
    top_k_recall = 40
    alpha = 0.5  # BM25 vs text embedding
    gamma = 0.5  # text hybrid vs ColPali in first stage
    beta = 0.5   # rerank vs ColPali in second stage

    print(f"\nParameters:")
    print(f"  alpha = {alpha} (BM25 vs Text Embedding)")
    print(f"  gamma = {gamma} (Text Hybrid vs ColPali)")
    print(f"  beta  = {beta} (Rerank vs ColPali)")
    print(f"  top_k_recall = {top_k_recall}")
    print("=" * 60)

    # Evaluate on easy
    print("\n[EASY TEST SET]")
    easy_metrics = evaluate_retrieval(
        easy_tests, chunks, chunk_ids, bm25, emb, embed_model,
        colpali_model, colpali_processor, colpali_page_embeds, reranker,
        top_k_values, top_k_recall, alpha, gamma, beta
    )
    print(f"  MRR:       {easy_metrics['mrr']:.4f}")
    for k in top_k_values:
        print(f"  Hit@{k}:    {easy_metrics[f'hit@{k}']:.4f}  |  Recall@{k}: {easy_metrics[f'recall@{k}']:.4f}")

    # Evaluate on hard
    print("\n[HARD TEST SET]")
    hard_metrics = evaluate_retrieval(
        hard_tests, chunks, chunk_ids, bm25, emb, embed_model,
        colpali_model, colpali_processor, colpali_page_embeds, reranker,
        top_k_values, top_k_recall, alpha, gamma, beta
    )
    print(f"  MRR:       {hard_metrics['mrr']:.4f}")
    for k in top_k_values:
        print(f"  Hit@{k}:    {hard_metrics[f'hit@{k}']:.4f}  |  Recall@{k}: {hard_metrics[f'recall@{k}']:.4f}")


if __name__ == "__main__":
    main()
