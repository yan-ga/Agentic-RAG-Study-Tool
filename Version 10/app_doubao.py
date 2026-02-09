import os, io, json, base64, uuid, sqlite3
import numpy as np
import requests
from PIL import Image

import streamlit as st

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.agents.middleware import (
    SummarizationMiddleware,
    ContextEditingMiddleware,
    ClearToolUsesEdit
)

DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DOUBAO_API_KEY = "d9676802-500b-49bc-a84f-2ee3373aa279"
DOUBAO_MODEL = "doubao-seed-1-6-lite-251015"
EMBED_URL = "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal"
EMBED_MODEL = "doubao-embedding-vision-250615"

# Utilities
def encode_image(image_path):
    img = Image.open(image_path)
    img.thumbnail((896, 896))
    buffered = io.BytesIO()
    img.convert("RGB").save(buffered, format="JPEG", quality=60)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Thread persistence
THREADS_FILE = "../memory state/threads.json"
SQLITE_PATH = "../memory state/checkpoints.sqlite"
def load_threads():
    if os.path.exists(THREADS_FILE):
        with open(THREADS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_threads(threads):
    with open(THREADS_FILE, "w", encoding="utf-8") as f:
        json.dump(threads, f, indent=2, ensure_ascii=False)

def get_or_create_thread(threads, name):
    if name not in threads:
        threads[name] = str(uuid.uuid4())
    return threads[name]

def delete_thread_from_sqlite(thread_id):
    with sqlite3.connect(SQLITE_PATH) as conn:
        conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        conn.commit()

# Cached loaders
@st.cache_data
def load_chunks(chunks_jsonl):
    chunks = []
    with open(chunks_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks

@st.cache_resource
def load_doubao_page_embeds(path):
    return np.load(path)

@st.cache_resource
def load_doubao_model():
    return ChatOpenAI(base_url=DOUBAO_BASE_URL, api_key=DOUBAO_API_KEY, model=DOUBAO_MODEL, temperature=0.0)

# Retrieval
def embed_query(text):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DOUBAO_API_KEY}",
    }
    payload = {
        "model": EMBED_MODEL,
        "input": [{"type": "text", "text": text}],
    }
    resp = requests.post(EMBED_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return np.array(resp.json()["data"]["embedding"], dtype=np.float32)

def retrieve(chunks, page_embeds, query, top_k=8):
    q_emb = embed_query(query)
    # Cosine similarity
    norms = np.linalg.norm(page_embeds, axis=1) * np.linalg.norm(q_emb)
    scores = page_embeds @ q_emb / np.where(norms > 0, norms, 1.0)
    order = np.argsort(-scores)[:top_k]
    return [chunks[i] for i in order]

# Load once
CHUNKS_PATH = "../data/new_chunks.jsonl"

chunks = load_chunks(CHUNKS_PATH)
page_embeds = load_doubao_page_embeds("../data/doubao_page_emb.npy")

doubao_model = load_doubao_model()

# Tools
@tool
def search_sources(query, top_k=8):
    """
    Retrieve top lecture chunks relevant to the query.

    Args:
        query: The search keywords for lecture content.
        top_k: The number of results to return. You must ALWAYS set this to 8 to ensure enough context for a high-quality answer.

    Returns:
        A JSON string containing a list of {chunk_id, doc_id, page_no, text, visual_summary, image_path}.
    """
    cands = retrieve(chunks, page_embeds, query, top_k)

    results = [
        {
            "chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "page_no": int(c["page_no"]),
            "text": (c.get("text") or "")[:800],
            "visual_summary": (c.get("visual_summary") or "")[:800],
            "image_path": c["image_path"],
        }
        for c in cands
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

    vision_analysis = doubao_model.invoke([msg]).content
    return f"\n--- DETAILED ANALYSIS OF {image_path} ---\n{vision_analysis}\n"

def run_agent(query, thread_id):
    model = load_doubao_model()

    system_prompt = (
        "You are a UNSW COMP9517 study assistant.\n\n"

        "For lecture content related question:\n"
        "1) Call 'search_sources' first to find the best slide(s).\n"
        "2) Use ONLY the returned text + visual summaries to decide which slide(s) are relevant.\n"
        "3) You should call 'analyse_slide_visuals' for ONLY the important slides you have identified as relevant based on their text and visual summaries.\n"
        "4) If multiple slides are relevant, you MUST fetch them ONE BY ONE. Step-by-step Loop:\n"
        "   a) Pick the single most relevant image_path.\n"
        "   b) Call 'analyse_slide_visuals' for ONLY that path.\n"
        "   c) Only then, call 'analyse_slide_visuals' for the NEXT path if needed.\n"
        "5) Answer ONLY using the evidence from the returned sources.\n"
        "6) Answer directly, clearly and structurally (short headings/sections are preferred).\n"
        "7) If evidence is missing/unclear, say what's missing and ask a focused question.\n"
        "8) Cite inline like [lectureX p.Y] for all sources you have used using doc_id + page_no from the tool results. Do not invent citations.\n"
    )

    if "checkpointer" not in st.session_state:
        cm = SqliteSaver.from_conn_string(SQLITE_PATH)
        st.session_state._checkpointer_cm = cm
        st.session_state.checkpointer = cm.__enter__()

    if "agent" not in st.session_state:
        st.session_state.agent = create_agent(
            model=model,
            tools=[search_sources, analyse_slide_visuals],
            system_prompt=system_prompt,
            checkpointer=st.session_state.checkpointer,
            middleware=[
                ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=8000, keep=4)]),
                SummarizationMiddleware(model=model, trigger=("tokens", 8000), keep=("messages", 24))
            ]
        )

    agent = st.session_state.agent
    result = agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": thread_id}}
    )

    return result["messages"][-1].content

# UI
st.title("COMP9517 Lecture Explainer")

with st.sidebar:
    st.header("Conversations")

    threads = load_threads()
    if "default" not in threads:
        threads["default"] = str(uuid.uuid4())
        save_threads(threads)

    if "active_thread" not in st.session_state:
        st.session_state.active_thread = "default"

    threads = load_threads()
    names = sorted(threads.keys())
    # We might delete the active thread in another session. Thus, we must check it.
    if st.session_state.active_thread not in threads:
        st.session_state.active_thread = "default"

    active = st.selectbox("Active conversation", names, index=names.index(st.session_state.active_thread))
    st.session_state.active_thread = active
    active_thread_id = threads[active]
    st.caption(f"thread_id: {active_thread_id}")

    new_name = st.text_input("New conversation name", value="", placeholder="e.g., lec7-midterm")
    if st.button("Create & switch", use_container_width=True) and new_name.strip():
        new_name = new_name.strip()
        threads = load_threads()
        if new_name in threads:
            st.warning("Thread already exists.")
        else:
            threads[new_name] = str(uuid.uuid4())
            save_threads(threads)
            st.session_state.active_thread = new_name
            st.success(f"Created '{new_name}'.")
            st.rerun()

    deletables = [n for n in names if n != "default"]
    delete_name = st.selectbox("Delete conversation", deletables) if deletables else None
    if st.button("Delete selected", use_container_width=True) and delete_name:
        threads = load_threads()
        tid = threads.pop(delete_name)
        save_threads(threads)
        delete_thread_from_sqlite(tid)
        if st.session_state.active_thread == delete_name:
            st.session_state.active_thread = "default"
        st.success(f"Deleted '{delete_name}'.")
        st.rerun()

question = st.text_input("Question", value="", placeholder="Ask anything from the COMP9517 slides…")

if st.button("Ask") and question.strip():
    threads = load_threads()
    active = st.session_state.active_thread
    thread_id = threads.get(active)

    if thread_id is None:
        st.session_state.active_thread = "default"
        st.error("This conversation no longer exists. Switched to default.")
        st.stop()

    with st.spinner("Thinking..."):
        answer = run_agent(question.strip(), thread_id)

    st.subheader("Answer")
    st.write(answer)
