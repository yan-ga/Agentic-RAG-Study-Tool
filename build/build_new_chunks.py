import os, json, re, fitz, base64, io
from PIL import Image

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

def encode_image(image_path):
    img = Image.open(image_path)
    img.thumbnail((896, 896)) 
    
    buffered = io.BytesIO()
    img.convert("RGB").save(buffered, format="JPEG", quality=60) 
    
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def clean_slide_text(s):
    s = s.replace("\u00ad", "")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def make_visual_summary(model, image_path):
    prompt = (
        "Describe the slide image literally for retrieval.\n"
        "Rules:\n"
        "- Describe this slide in 2-4 concise sentences for a search engine.\n"
        "- Mention the type of visual and key objects.\n"
        "- Do NOT guess context beyond what is visible.\n"
        "- Do NOT describe footers, slide numbers, or university logos.\n"
    )
    
    b64_data = encode_image(image_path)

    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}}
    ])

    res = model.invoke([msg])
    return (res.content or "").strip()


def main(in_dir, out_jsonl, images_dir):
    processed_ids = set()
    if os.path.exists(out_jsonl):
        with open(out_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line_data = json.loads(line)
                processed_ids.add(line_data["chunk_id"])
        print(f"Found {len(processed_ids)} existing records. Resuming...")
    
    pdfs = sorted([f for f in os.listdir(in_dir) if f.lower().endswith(".pdf")])

    model = ChatOllama(
        model="qwen3-vl:4b",
        temperature=0.0,
        num_ctx=10000,
        timeout=120
    )

    n_written = 0
    with open(out_jsonl, "a", encoding="utf-8") as out:
        for fname in pdfs:
            pdf_path = os.path.join(in_dir, fname)
            doc_id = os.path.splitext(fname)[0]

            doc = fitz.open(pdf_path)
            for i in range(doc.page_count):
                page_no = i + 1
                
                chunk_id = f"{doc_id}_p{page_no:03d}"
                if chunk_id in processed_ids:
                    continue
                
                if page_no % 10 == 0 or page_no == 1 or page_no == doc.page_count:
                    print(f"[{doc_id}] {page_no}/{doc.page_count} pages | written={n_written}")
        
                page = doc.load_page(i)

                text = clean_slide_text(page.get_text("text") or "")

                img_name = f"{doc_id}_p{page_no:03d}.png"
                img_path = os.path.join(images_dir, img_name)

                visual_summary = make_visual_summary(model, img_path)

                retrieval_text = (text + "\n\n[VISUAL_SUMMARY]\n" + visual_summary).strip()

                rec = {
                    "doc_id": doc_id,
                    "source_path": pdf_path.replace("\\", "/"),
                    "page_no": page_no,
                    "chunk_id": f"{doc_id}_p{page_no:03d}",
                    "text": text,
                    "visual_summary": visual_summary,
                    "retrieval_text": retrieval_text,
                    "image_path": img_path.replace("\\", "/"),
                }

                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                n_written += 1

            doc.close()

    print(f"Wrote {n_written} chunks to: {out_jsonl}")


if __name__ == "__main__":
    in_dir = "data/raw_pdfs"
    out_jsonl = "data/new_chunks.jsonl"
    images_dir = "data/page_images"

    main(in_dir, out_jsonl, images_dir)
