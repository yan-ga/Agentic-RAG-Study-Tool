import os
import json
import re
import sys
import fitz

def clean_slide_text(s):
    s = s.replace("\u00ad", "")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()

def main(in_dir, out_jsonl):
    pdfs = [f for f in os.listdir(in_dir) if f.lower().endswith(".pdf")]
    pdfs.sort()

    with open(out_jsonl, "w", encoding="utf-8") as out:
        for fname in pdfs:
            pdf_path = os.path.join(in_dir, fname)
            doc_id = os.path.splitext(fname)[0]
            doc_title = doc_id

            doc = fitz.open(pdf_path)
            for i in range(doc.page_count):
                page_no = i + 1
                page = doc.load_page(i)

                text = clean_slide_text(page.get_text("text") or "")
                if not text:
                    continue

                rec = {
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "source_path": pdf_path.replace("\\", "/"),
                    "page_start": page_no,
                    "page_end": page_no,
                    "chunk_id": f"{doc_id}_p{page_no:03d}",
                    "text": text,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")

            doc.close()

    print(f"Wrote chunks to: {out_jsonl}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
