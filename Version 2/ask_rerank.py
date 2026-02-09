import sys, os, json, re, subprocess
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

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

def build_prompt(question, sources):
    blocks = []
    for s in sources:
        blocks.append(f"[{s['doc_id']} p.{s['page_start']}]\n{s['text'].strip()}")
    ctx = "\n\n".join(blocks)
    return (
        "You are a study assistant. Answer using ONLY the provided sources.\n"
        "If the sources do not contain the answer, say you don't have enough information.\n"
        "Make sure you cite sources inline like [lecture17 p.22]. Do not invent citations.\n\n"
        f"Question:\n{question}\n\nSources:\n{ctx}\n"
    )

def ollama(prompt, model="qwen2.5:7b-instruct"):
    p = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return p.stdout.decode("utf-8", errors="ignore")


def retrieve_hybrid_candidates(chunks, emb, embed_model, query, top_k_recall=40, alpha=0.5):
    # BM25 stage
    corpus_tokens = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    bm25_scores = np.array(bm25.get_scores(tokenize(query)), dtype=np.float32)

    # Embeddings stage
    q = embed_model.encode([query], normalize_embeddings=True)[0]
    emb_scores = (emb @ q).astype(np.float32)

    # Hybrid score
    final = alpha * minmax(bm25_scores) + (1 - alpha) * minmax(emb_scores)
    ranked = np.argsort(-final)[:top_k_recall]
    candidates = [chunks[i] for i in ranked]
    return candidates

def rerank_cross_encoder(reranker, query, candidates, top_k_final=8):
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    order = np.argsort(-scores)[:top_k_final]
    reranked = [candidates[i] for i in order]
    return reranked

def main(chunks_jsonl, embeddings_npy, question, llm_model="qwen2.5:7b-instruct",
         top_k_recall=40, top_k_final=8, alpha=0.5):

    chunks = load_chunks(chunks_jsonl)
    emb = np.load(embeddings_npy)

    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    candidates = retrieve_hybrid_candidates(
        chunks, emb, embed_model, question,
        top_k_recall=top_k_recall,
        alpha=alpha
    )

    sources = rerank_cross_encoder(reranker, question, candidates, top_k_final=top_k_final)
    prompt = build_prompt(question, sources)
    
    answer = ollama(prompt, model=llm_model)
    print(answer)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main(sys.argv[1], sys.argv[2], " ".join(sys.argv[3:]))
