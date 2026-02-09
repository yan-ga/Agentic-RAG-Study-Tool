import json, sys, re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def load_chunks(chunks_jsonl):
    chunks = []
    with open(chunks_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks

def minmax(x):
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)

def main(chunks_jsonl, embeddings_npy, query, top_k=8, alpha=0.5):
    chunks = load_chunks(chunks_jsonl)

    # BM25 scores
    corpus_tokens = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    bm25_scores = np.array(bm25.get_scores(tokenize(query)), dtype=np.float32)

    # Embedding scores
    emb = np.load(embeddings_npy)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    q = model.encode([query], normalize_embeddings=True)[0]
    embed_scores = (emb @ q).astype(np.float32)

    # Combine
    bm25_n = minmax(bm25_scores)
    emb_n  = minmax(embed_scores)
    final = alpha * bm25_n + (1 - alpha) * emb_n
    ranked = np.argsort(-final)[:top_k]

    for rank, i in enumerate(ranked, 1):
        c = chunks[i]
        preview = c["text"].replace("\n", " ")
        preview = preview[:200] + ("..." if len(preview) > 200 else "")
        print(f"{rank}. {c['doc_id']} p.{c['page_start']} final={final[i]:.3f} bm25={bm25_scores[i]:.2f} emb={embed_scores[i]:.3f}")
        print(f"   {preview}\n")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], " ".join(sys.argv[3:]))
