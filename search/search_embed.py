import json, sys
import numpy as np
from sentence_transformers import SentenceTransformer

def load_chunks(chunks_jsonl):
    chunks = []
    with open(chunks_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks

def main(chunks_jsonl, embeddings_npy, query, top_k=8):
    chunks = load_chunks(chunks_jsonl)
    emb = np.load(embeddings_npy)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    q = model.encode([query], normalize_embeddings=True)[0]

    # cosine similarity since embeddings are normalised => dot product
    scores = emb @ q
    ranked = np.argsort(-scores)[:top_k]

    for rank, i in enumerate(ranked, 1):
        c = chunks[i]
        preview = c["text"].replace("\n", " ")
        preview = preview[:200] + ("..." if len(preview) > 200 else "")
        print(f"{rank}. {c['doc_id']} p.{c['page_start']} score={float(scores[i]):.3f}")
        print(f"   {preview}\n")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], " ".join(sys.argv[3:]))
