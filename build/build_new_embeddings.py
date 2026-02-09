import json
import numpy as np
from sentence_transformers import SentenceTransformer

def load_chunks(chunks_jsonl):
    chunks = []
    with open(chunks_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks

def main(chunks_jsonl, out_npy):
    chunks = load_chunks(chunks_jsonl)

    texts = [c["retrieval_text"] for c in chunks]

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    np.save(out_npy, emb)
    print(f"Saved embeddings: {out_npy} shape={emb.shape}")

if __name__ == "__main__":
    chunks_jsonl = "data/new_chunks.jsonl"
    out_npy = "data/new_embeddings.npy"
    main(chunks_jsonl, out_npy)
