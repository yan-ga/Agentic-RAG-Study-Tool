import json
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

def load_pages(pages_jsonl):
    pages = []
    with open(pages_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pages.append(json.loads(line))
    return pages

def grid_tiles(img, grid):
    w, h = img.size
    tiles = []
    for r in range(grid):
        y0 = (h * r) // grid
        y1 = (h * (r + 1)) // grid
        for c in range(grid):
            x0 = (w * c) // grid
            x1 = (w * (c + 1)) // grid
            tiles.append(img.crop((x0, y0, x1, y1)))
    return tiles

def main(pages_jsonl, out_npy, out_meta, grid):
    pages = load_pages(pages_jsonl)
    clip = SentenceTransformer("sentence-transformers/clip-ViT-B-32")

    images = []
    meta = []

    for p in pages:
        img_path = p["image_path"]
        base = Image.open(img_path).convert("RGB")
        tiles = grid_tiles(base, grid)

        for t_idx, tile in enumerate(tiles):
            images.append(tile)
            meta.append({
                "doc_id": p["doc_id"],
                "page_start": p["page_start"],
                "tile_idx": t_idx,
                "image_path": img_path,
            })

    emb = clip.encode(
        images,
        batch_size=32,
        normalize_embeddings=True,
    )
    emb = np.asarray(emb, dtype=np.float32)
    np.save(out_npy, emb)

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved CLIP embeddings: {out_npy} shape={emb.shape}")
    print(f"Saved CLIP meta: {out_meta} items={len(meta)} grid={grid}")

if __name__ == "__main__":
    pages_jsonl = "data/pages.jsonl"
    out_npy = "data/clip_embeddings.npy"
    out_meta = "data/clip_meta.json"
    grid = 5

    main(pages_jsonl, out_npy, out_meta, grid=grid)
