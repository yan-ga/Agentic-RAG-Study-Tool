import sys, json, re, base64
import numpy as np
import io
import torch
from PIL import Image

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

# Load ../data/models
chunks = load_chunks("../data/new_chunks.jsonl")
emb = np.load("../data/new_embeddings.npy")

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
bm25 = BM25Okapi([tokenize(c["retrieval_text"]) for c in chunks])

# ColPali model + processor + precomputed page embeddings
COLPALI_NAME = "vidore/colpali-v1.3-hf"
colpali_model = ColPaliForRetrieval.from_pretrained(COLPALI_NAME, dtype=torch.float32, device_map="cpu")
colpali_processor = ColPaliProcessor.from_pretrained(COLPALI_NAME)
colpali_page_embeds = torch.load("../data/colpali_page_emb.pt")

# Vision model
vision_model = ChatOllama(model="qwen3-vl:4b", temperature=0.0, num_ctx=10000, timeout=120)

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

    print("\n" + "=" * 80)
    print(f"[DEBUG] Query: {query} | Beta used: {beta}")
    print("[DEBUG] Top-8 retrieved pages:\n")
    for rank, c in enumerate(top, start=1):
        doc_id = c["doc_id"]
        p = int(c["page_no"])
        img = f"../data/page_images/{doc_id}_p{p:03d}.png"
        print(f"{rank:02d}. {doc_id} p.{p:03d} | {img}")
    print("=" * 80 + "\n")

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
    
    print(f"[QUERY] {query}")
    print(f"[ANALYZING PIXELS] {image_path}")

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

def main(query):
    model = ChatOllama(model="qwen3:4b", temperature=0.0, num_ctx=10000)

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
    print(result["messages"][-1].content)

if __name__ == "__main__":
    main(" ".join(sys.argv[1:]))
