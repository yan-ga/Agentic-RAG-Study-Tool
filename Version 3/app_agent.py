import json, re
import numpy as np
import streamlit as st

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama

# Utilities
def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def minmax(x):
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)

@st.cache_data
def load_chunks(chunks_jsonl):
    chunks = []
    with open(chunks_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks

@st.cache_data
def load_embeddings(path):
    return np.load(path)

@st.cache_resource
def load_embed_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

@st.cache_resource
def build_bm25(chunks):
    return BM25Okapi([tokenize(c["text"]) for c in chunks])

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

def extract_tool_sources(messages, tool_name="retrieve_sources"):
    sources = []
    seen = set()

    for m in messages:
        if getattr(m, "name", None) != tool_name:
            continue

        items = json.loads(m.content)
        for it in items:
            key = (it["doc_id"], it["page_start"], (it["text"])[:120])
            if key in seen:
                continue
            seen.add(key)
            sources.append(it)

    return sources

# Streamlit UI
st.title("COMP9517 Lecture Explainer")

chunks_path = "../data/chunks.jsonl"
emb_path = "../data/embeddings.npy"
model_name = "qwen2.5:7b-instruct"

TOP_K_RECALL = 40
TOP_K_FINAL = 8
ALPHA = 0.5
TEMPERATURE = 0.0

question = st.text_input("Question", value="", placeholder="Ask anything from the COMP9517 slides…")

if st.button("Ask"):
    q = question.strip()

    chunks = load_chunks(chunks_path)
    emb = load_embeddings(emb_path)
    embed_model = load_embed_model()
    reranker = load_reranker()
    bm25 = build_bm25(chunks)

    # Tool the agent can call
    @tool
    def retrieve_sources(query):
        """Retrieve top lecture chunks relevant to the query. Returns JSON list of {doc_id, page_start, text}."""
        cands = retrieve_hybrid(chunks, bm25, emb, embed_model, query=query, top_k=TOP_K_RECALL, alpha=ALPHA)
        top = rerank(reranker, question=query, candidates=cands, top_k=TOP_K_FINAL)

        out = []
        for c in top:
            out.append({
                "doc_id": c["doc_id"],
                "page_start": c["page_start"],
                "text": (c["text"])[:800],
            })
        return json.dumps(out, ensure_ascii=False)
    
    system_prompt = (
        "You are a COMP9517 study assistant.\n"
        "Use tools when they improve accuracy or reduce guessing.\n"
        "- For lecture-specific questions, retrieve evidence before answering. Base your claims on the lecture slides/provided sources.\n" 
        "- You should explain and expand ideas, but do not add unsupported lecture-specific details.\n"
        "- Answer fully, clearly and structurally (short headings/sections are preferred).\n"
        "- Do not invent claims. If evidence is missing or ambiguous, say what’s missing and ask a focused question.\n"
        "- You should cite sources inline like [lectureX p.Y] for each source you have used. Do not invent citations.\n"
    )

    model = ChatOllama(model=model_name, temperature=TEMPERATURE)
    agent = create_agent(model=model, tools=[retrieve_sources], system_prompt=system_prompt)

    with st.spinner("Thinking..."):
        result = agent.invoke({"messages": [{"role": "user", "content": q}]})

    messages = result["messages"]
    answer = messages[-1].content

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    sources = extract_tool_sources(messages, tool_name="retrieve_sources")
    for s in sources:
        st.markdown(f"**{s['doc_id']} p.{s['page_start']}**")
        st.write(s["text"][:800] + ("..." if len(s["text"]) > 800 else ""))
        st.divider()
