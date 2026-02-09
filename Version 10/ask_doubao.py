import os, io, json, base64, uuid, sqlite3
import numpy as np
import requests
from PIL import Image

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

def load_chunks(chunks_jsonl):
    chunks = []
    with open(chunks_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks

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

# Thread Management
THREADS_FILE = "../memory state/threads.json"
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
    with sqlite3.connect("../memory state/checkpoints.sqlite") as conn:
        conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        conn.commit()

# Load data
chunks = load_chunks("../data/new_chunks.jsonl")

# Precomputed Doubao page embeddings
page_embeds = np.load("../data/doubao_page_emb.npy")

# Vision model (Doubao supports vision natively)
vision_model = ChatOpenAI(base_url=DOUBAO_BASE_URL, api_key=DOUBAO_API_KEY, model=DOUBAO_MODEL, temperature=0.0)

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

    print("\n" + "=" * 80)
    print(f"[DEBUG] Query: {query}")
    print("[DEBUG] Top-8 retrieved pages:\n")
    for rank, c in enumerate(cands, start=1):
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

def main():
    model = ChatOpenAI(base_url=DOUBAO_BASE_URL, api_key=DOUBAO_API_KEY, model=DOUBAO_MODEL, temperature=0.0)

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

    with SqliteSaver.from_conn_string("../memory state/checkpoints.sqlite") as checkpointer:
        agent = create_agent(
            model=model,
            tools=[search_sources, analyse_slide_visuals],
            system_prompt=system_prompt,
            checkpointer=checkpointer,
            middleware=[
                ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=8000, keep=4)]),
                SummarizationMiddleware(model=model, trigger=("tokens", 8000), keep=("messages", 24))
            ]
        )

        threads = load_threads()
        active = "default"
        thread_id = get_or_create_thread(threads, active)
        save_threads(threads)

        print("Ask anything from the COMP9517 slides…")
        print("Type /help for commands.\n")
        while True:
            try:
                query = input(f"You[{active}]> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not query:
                print("Query cannot be empty.")
                continue

            if query == "/help":
                print(
                    "\nCommands:\n"
                    "  /help                  Show this help message\n"
                    "  /threads               List all threads\n"
                    "  /active                Show the current active thread\n"
                    "  /switch <name>         Switch to an existing thread\n"
                    "  /new <name>            Create a new thread and switch to it\n"
                    "  /delete <name>         Delete a thread\n"
                    "  /exit                  Exit the program\n"
                )
                continue

            if query == "/threads":
                print("Threads:", ", ".join(sorted(threads.keys())))
                print("Active:", active)
                continue

            if query == "/active":
                print("Active:", active)
                continue

            if query.startswith("/switch "):
                name = query.split(" ", 1)[1].strip()
                if not name:
                    print("Usage: /switch <name>")
                    continue
                if name not in threads:
                    print(f"No such thread '{name}'. Use /new {name} to create it.")
                    continue
                active = name
                thread_id = threads[active]
                print(f"Switched to '{active}'.")
                continue

            if query.startswith("/new "):
                name = query.split(" ", 1)[1].strip()
                if not name:
                    print("Usage: /new <name>")
                    continue
                if name in threads:
                    print(f"Thread '{name}' already exists. Use /switch {name} or choose another name.")
                    continue

                threads[name] = str(uuid.uuid4())
                save_threads(threads)
                active = name
                thread_id = threads[active]
                print(f"Created and switched to '{active}'.")
                continue

            if query.startswith("/delete "):
                name = query.split(" ", 1)[1].strip()
                if not name:
                    print("Usage: /delete <name>")
                    continue
                if name not in threads:
                    print(f"No such thread '{name}'.")
                    continue
                if name == "default":
                    print("Refusing to delete 'default'.")
                    continue

                tid = threads.pop(name)
                save_threads(threads)
                delete_thread_from_sqlite(tid)
                print(f"Deleted '{name}'.")

                if active == name:
                    active = "default"
                    thread_id = threads["default"]
                    print("Switched to 'default'.")

                continue

            if query == "/exit":
                break

            result = agent.invoke(
                {"messages": [HumanMessage(content=query)]},
                config={"configurable": {"thread_id": thread_id}}
            )
            print(result["messages"][-1].content)
            print()

if __name__ == "__main__":
    main()
