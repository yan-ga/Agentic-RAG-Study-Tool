import os, sys, json
import fitz

def main(in_dir, out_pages_jsonl, out_img_dir, zoom=2.0):
    os.makedirs(out_img_dir, exist_ok=True)

    pdfs = [f for f in os.listdir(in_dir) if f.lower().endswith(".pdf")]
    pdfs.sort()

    with open(out_pages_jsonl, "w", encoding="utf-8") as out:
        for fname in pdfs:
            pdf_path = os.path.join(in_dir, fname)
            doc_id = os.path.splitext(fname)[0]

            doc = fitz.open(pdf_path)
            for i in range(doc.page_count):
                page_no = i + 1
                page = doc.load_page(i)

                # Render page to PNG
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)

                img_name = f"{doc_id}_p{page_no:03d}.png"
                img_path = os.path.join(out_img_dir, img_name)
                pix.save(img_path)

                rec = {
                    "doc_id": doc_id,
                    "page_start": page_no,
                    "image_path": img_path.replace("\\", "/"),
                    "source_path": pdf_path.replace("\\", "/"),
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")

            doc.close()

    print(f"Wrote pages to: {out_pages_jsonl}")
    print(f"Wrote images to: {out_img_dir}")

if __name__ == "__main__":
    zoom = float(sys.argv[4]) if len(sys.argv) >= 5 else 2.0
    main(sys.argv[1], sys.argv[2], sys.argv[3], zoom=zoom)
