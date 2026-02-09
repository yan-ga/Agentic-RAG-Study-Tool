import os, json
import torch
from PIL import Image
from transformers import ColPaliForRetrieval, ColPaliProcessor

MODEL_NAME = "vidore/colpali-v1.3-hf"

def main(image_dir="data/page_images", out_pt="data/colpali_page_emb.pt", out_meta="data/colpali_meta.json", batch_size=8):

    model = ColPaliForRetrieval.from_pretrained(MODEL_NAME, dtype=torch.float32, device_map="cpu")
    processor = ColPaliProcessor.from_pretrained(MODEL_NAME)

    def sort_key_png(fn):
        base = os.path.basename(fn)
        doc_id, page_part = base.rsplit("_p", 1)
        page_no = int(page_part.split(".")[0])
        return (doc_id, page_no)

    fns = [fn for fn in os.listdir(image_dir) if fn.lower().endswith(".png")]
    fns.sort(key=sort_key_png)
    paths = [os.path.join(image_dir, fn) for fn in fns]

    metas = []
    embs = []

    model.eval()
    with torch.no_grad():
        total = len(paths)
        for start in range(0, total, batch_size):
            batch_paths = paths[start:start + batch_size]

            print(f"[{start+1}/{total}] -> {os.path.basename(batch_paths[0])}")

            imgs = [Image.open(p).convert("RGB") for p in batch_paths]

            inputs = processor(images=imgs, return_tensors="pt").to(model.device)
            out = model(**inputs).embeddings

            for i, p in enumerate(batch_paths):
                embs.append(out[i].to("cpu"))

                base = os.path.basename(p)
                doc_id, page_part = base.rsplit("_p", 1)
                page_no = int(page_part.split(".")[0])
                metas.append({"doc_id": doc_id, "page_no": page_no, "image_path": p})

    torch.save(embs, out_pt)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(embs)} page embeddings -> {out_pt}")
    print(f"Saved meta -> {out_meta}")

if __name__ == "__main__":
    main()
