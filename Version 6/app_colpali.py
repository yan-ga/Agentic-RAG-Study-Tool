import json, re, base64, io
import numpy as np
import torch
from PIL import Image

import streamlit as st

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain.tools import tool
from langchain_ollama import ChatOllama

from transformers import ColPaliForRetrieval, ColPaliProcessor

# Utilities
def encode_image(image_path):
    img = Image.open(image_path)
    img.thumbnail((896, 896))
    buffered = io.BytesIO()
    img.convert("RGB").save(buffered, format="JPEG", quality=60)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

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
def load_bm25(chunks):
    return BM25Okapi([tokenize(c["retrieval_text"]) for c in chunks])

@st.cache_resource
def load_colpali():
    model = ColPaliForRetrieval.from_pretrained("vidore/colpali-v1.3-hf", dtype=torch.float32, device_map="cpu")
    processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.3-hf")
    return model, processor

@st.cache_resource
def load_colpali_page_embeds(path):
    return torch.load(path)

@st.cache_resource
def load_vision_model():
    return ChatOllama(model="qwen3-vl:4b", temperature=0.0, num_ctx=10000, timeout=120)

@st.cache_resource
def load_text_agent_model():
    return ChatOllama(model="qwen3:4b", temperature=0.0, num_ctx=10000)

# Hybrid retrieval
def retrieve(chunks, bm25, emb, embed_model, colpali_model, colpali_processor, colpali_page_embeds, query, top_k=40, alpha=0.5, gamma=0.5):
    # BM25 stage
    bm25_scores = np.array(bm25.get_scores(tokenize(query)), dtype=np.float32)

    # Text-embeddings stage
    q_emb = embed_model.encode([query], normalize_embeddings=True)[0]
    emb_scores = (emb @ q_emb).astype(np.float32)

    # ColPali visual stage
    colpali_model.eval()
    with torch.no_grad():
        q_inputs = colpali_processor(text=[query], return_tensors="pt").to(colpali_model.device)
        q_emb = colpali_model(**q_inputs).embeddings
        colpali_scores = colpali_processor.score_retrieval(q_emb, colpali_page_embeds)
        colpali_scores = colpali_scores[0].cpu().numpy().astype(np.float32)

    # Fuse scores
    text_hybrid = alpha * minmax(bm25_scores) + (1 - alpha) * minmax(emb_scores)
    final_scores = (1 - gamma) * text_hybrid + gamma * minmax(colpali_scores)

    order = np.argsort(-final_scores)[:top_k]
    return [chunks[i] for i in order], colpali_scores, order

# Load once
CHUNKS_PATH = "../data/new_chunks.jsonl"
EMB_PATH = "../data/new_embeddings.npy"
COLPALI_EMB_PATH = "../data/colpali_page_emb.pt"

chunks = load_chunks(CHUNKS_PATH)
emb = load_embeddings(EMB_PATH)

embed_model = load_embed_model()
reranker = load_reranker()
bm25 = load_bm25(chunks)

colpali_model, colpali_processor = load_colpali()
colpali_page_embeds = load_colpali_page_embeds(COLPALI_EMB_PATH)

vision_model = load_vision_model()

# Tools
@tool
def search_sources(query, beta=0.5, top_k=8):
    """
    Retrieve top lecture chunks relevant to the query.
    
    Args:
        query: The search keywords for lecture content.
        beta: Rerank weight (0.0 to 1.0) for the ColPali visual signal.
                - Use a HIGH beta (e.g. 0.8-0.9) when the user is asking to find a specific image, diagram, figure, architecture, 
                or any visual content.
                - Use a LOW beta (e.g. 0.1-0.2) when the user is asking for definitions, formulas, or “explain” style questions 
                where the slide text matters most.
                - Use a MID beta (e.g. 0.4-0.6) when the user’s question is general and you want a balanced search that uses both 
                text and visuals.
        top_k: The number of results to return. You must ALWAYS set this to 8 to ensure enough context for a high-quality answer.
        
    Returns:
        A JSON string containing a list of {chunk_id, doc_id, page_no, text, visual_summary, image_path}.
    """
    cands, colpali_scores, cand_indices = retrieve(chunks, bm25, emb, embed_model, colpali_model, colpali_processor, colpali_page_embeds, query)

    # Cross-encoder rerank on text
    pairs = [(query, c["retrieval_text"]) for c in cands]
    rerank_scores = reranker.predict(pairs, batch_size=16).astype(np.float32)

    # Pull ColPali scores for these candidates
    c_colpali_scores = colpali_scores[cand_indices].astype(np.float32)

    final_scores = (1 - beta) * minmax(rerank_scores) + beta * minmax(c_colpali_scores)
    order = np.argsort(-final_scores)[:top_k]
    top = [cands[i] for i in order]

    results = [
        {
            "chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "page_no": int(c["page_no"]),
            "text": (c.get("text") or "")[:800],
            "visual_summary": (c.get("visual_summary") or "")[:800],
            "image_path": c["image_path"],
        }
        for c in top
    ]
    return json.dumps(results, ensure_ascii=False)

@tool
def analyse_slide_visuals(query, image_path):
    """
    Analyses the slide at image_path to answer a specific query.
    
    Args:
        query: The specific question or detail to look for on the slide.
        image_path: The file path to the slide image (from search_sources).
        
    Returns:
        A detailed text analysis of the slide's contents.
    """

    b64_data = encode_image(image_path)
    vision_prompt = (
        f"The user is asking: '{query}'.\n"
        "Carefully examine this lecture slide and provide a detailed analysis that DIRECTLY addresses that question.\n"
    )

    msg = HumanMessage(content=[
        {"type": "text", "text": vision_prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}},
    ])

    vision_analysis = vision_model.invoke([msg]).content
    return f"\n--- DETAILED ANALYSIS OF {image_path} ---\n{vision_analysis}\n"

def run_agent(query):
    model = load_text_agent_model()

    system_prompt = (
        "You are a UNSW COMP9517 study assistant.\n\n"
        
        "For lecture content related question:"
        "1) Call 'search_sources' first to find the best slide(s).\n"
        "2) Use ONLY the returned text + visual summaries to decide which slide(s) are relevant.\n"
        "3) You should call 'analyse_slide_visuals' for ONLY the important slides you have identified as relevant based on their text and visual summaries.\n"
        "4) If multiple slides are relevant, you MUST fetch them ONE BY ONE. Step-by-step Loop:\n"
        "   a) Pick the single most relevant image_path.\n"
        "   b) Call 'analyse_slide_visuals' for ONLY that path.\n"
        "   c) Only then, call 'analyse_slide_visuals' for the NEXT path if needed.\n"
        "5) Answer ONLY using the evidence from the returned sources.\n"
        "6) Answer directly, clearly and structurally (short headings/sections are preferred).\n"
        "7) If evidence is missing/unclear, say what’s missing and ask a focused question.\n"
        "8) Cite inline like [lectureX p.Y] for all sources you have used using doc_id + page_no from the tool results. Do not invent citations.\n"
    )

    agent = create_agent(model=model, tools=[search_sources, analyse_slide_visuals], system_prompt=system_prompt)
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content

# UI
st.title("COMP9517 Lecture Explainer")
question = st.text_input("Question", value="", placeholder="Ask anything from the COMP9517 slides…")

if st.button("Ask") and question.strip():
    with st.spinner("Thinking..."):
        answer = run_agent(question.strip())

    st.subheader("Answer")
    st.write(answer)
