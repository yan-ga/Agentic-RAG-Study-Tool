import os, json, re, subprocess
import numpy as np
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

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
def build_bm25(chunks):
    corpus_tokens = [tokenize(c["text"]) for c in chunks]
    return BM25Okapi(corpus_tokens)

def minmax(x):
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)

def retrieve_hybrid(chunks, bm25, emb, embed_model, query, top_k=8, alpha=0.5):
    bm25_scores = np.array(bm25.get_scores(tokenize(query)), dtype=np.float32)

    # Embeddings
    q = embed_model.encode([query], normalize_embeddings=True)[0]
    emb_scores = (emb @ q).astype(np.float32)
    
    # Hybrid
    final = alpha * minmax(bm25_scores) + (1 - alpha) * minmax(emb_scores)
    ranked = np.argsort(-final)[:top_k]
    return [chunks[i] for i in ranked]


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

st.title("COMP9517 Lecture Explainer")

chunks_path = "../data/chunks.jsonl"
emb_path = "../data/embeddings.npy"
model_name = "qwen2.5:7b-instruct"

question = st.text_input("Question", value="", placeholder="Ask anything from the COMP9517 slides…")
if st.button("Ask"):
    chunks = load_chunks(chunks_path)
    emb = load_embeddings(emb_path)
    embed_model = load_embed_model()
    bm25 = build_bm25(chunks)

    sources = retrieve_hybrid(chunks, bm25, emb, embed_model, question, top_k=8, alpha=0.5)
    prompt = build_prompt(question, sources)
    answer = ollama(prompt, model=model_name)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for s in sources:
        st.markdown(f"**{s['doc_id']} p.{s['page_start']}**")
        st.write(s["text"][:800] + ("..." if len(s["text"]) > 800 else ""))
        st.divider()
