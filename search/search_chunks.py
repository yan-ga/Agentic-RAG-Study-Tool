import json
import sys
import re
import numpy as np
from rank_bm25 import BM25Okapi

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def load_chunks(chunks_jsonl):
    chunks = []
    with open(chunks_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks

def main(chunks_jsonl, query, top_k=8):
    chunks = load_chunks(chunks_jsonl)

    corpus_tokens = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)

    q_tokens = tokenize(query)
    scores = np.array(bm25.get_scores(q_tokens), dtype=np.float32)
    ranked = np.argsort(-scores)[:top_k]

    for rank, i in enumerate(ranked, 1):
        c = chunks[i]
        preview = c["text"].replace("\n", " ")
        preview = preview[:200] + ("..." if len(preview) > 200 else "")
        print(f"{rank}. {c['doc_id']} p.{c['page_start']} score={scores[i]:.3f}")
        print(f"   {preview}\n")

if __name__ == "__main__":
    chunks_jsonl = sys.argv[1]
    query = " ".join(sys.argv[2:])
    main(chunks_jsonl, query)
