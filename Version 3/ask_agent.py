import sys, json, re
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama

# Utilities
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

def retrieve_hybrid(chunks, bm25, emb, embed_model, query, top_k=40, alpha=0.5):
    # BM25 stage
    bm25_scores = np.array(bm25.get_scores(tokenize(query)), dtype=np.float32)
    
    # Embeddings stage
    q_emb = embed_model.encode([query], normalize_embeddings=True)[0]
    emb_scores = (emb @ q_emb).astype(np.float32)
    
    # Hybrid score
    final_scores = alpha * minmax(bm25_scores) + (1 - alpha) * minmax(emb_scores)
    order = np.argsort(-final_scores)[:top_k]
    return [chunks[i] for i in order]

def rerank(reranker, question, candidates, top_k=8):
    pairs = [(question, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    order = np.argsort(-scores)[:top_k]
    return [candidates[i] for i in order]

# Load existing ../data/models once
chunks = load_chunks("../data/chunks.jsonl")
emb = np.load("../data/embeddings.npy")

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

# Tool the agent can call
@tool
def retrieve_sources(query):
    """Retrieve top lecture chunks relevant to the query. Returns a JSON list of {doc_id, page_start, text}."""
    cands = retrieve_hybrid(chunks, bm25, emb, embed_model, query=query, top_k=40, alpha=0.5)
    top = rerank(reranker, question=query, candidates=cands, top_k=8)

    out = []
    for c in top:
        out.append({
            "doc_id": c["doc_id"],
            "page_start": c["page_start"],
            "text": (c["text"])[:800],
        })
    return json.dumps(out, ensure_ascii=False)

def main(query):
    model = ChatOllama(model="qwen2.5:7b-instruct", temperature=0.0)
    
    system_prompt = (
        "You are a COMP9517 study assistant. Answer using ONLY the lecture slides/provided sources.\n"
        "Use tools when they improve accuracy or reduce guessing.\n"
        "- For lecture-specific questions, retrieve evidence before answering. Base your claims on the lecture slides/provided sources.\n" 
        "- You should explain and expand ideas, but do not add unsupported lecture-specific details.\n"
        "- Answer fully, clearly and structurally (short headings/sections are preferred).\n"
        "- Do not invent claims. If evidence is missing or ambiguous, say what’s missing and ask a focused question.\n"
        "- You should cite sources inline like [lectureX p.Y] for each source you have used. Do not invent citations.\n"
    )

    agent = create_agent(model=model, tools=[retrieve_sources], system_prompt=system_prompt)
    
    # Sending the agent the current state of the conversation
    result = agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    
    # result is a dict (the agent "state"),
    # {
    #   "messages": [HumanMessage(...), AIMessage(...), ToolMessage(...), AIMessage(...final...)]
    # }
    # So result["messages"] is a list of message objects, and [-1] picks the last one (final answer).
    answer = result["messages"][-1].content
    print(answer)
    return answer

if __name__ == "__main__":
    main(" ".join(sys.argv[1:]))
