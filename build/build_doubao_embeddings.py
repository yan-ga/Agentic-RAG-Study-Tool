import os, io, json, time, base64
import numpy as np
import requests
from PIL import Image

DOUBAO_API_KEY = "REDACTED_API_KEY"
EMBED_URL = "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal"
EMBED_MODEL = "doubao-embedding-vision-250615"


def encode_image(path):
    img = Image.open(path)
    img.thumbnail((896, 896))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=60)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def sort_key_png(fn):
    base = os.path.basename(fn)
    doc_id, page_part = base.rsplit("_p", 1)
    page_no = int(page_part.split(".")[0])
    return (doc_id, page_no)


def embed_image(b64_data, max_retries=3):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DOUBAO_API_KEY}",
    }
    payload = {
        "model": EMBED_MODEL,
        "input": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}}],
    }
    for attempt in range(max_retries):
        resp = requests.post(EMBED_URL, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            return resp.json()["data"]["embedding"]
        print(f"  Attempt {attempt+1} failed ({resp.status_code}): {resp.text[:200]}", flush=True)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed after {max_retries} retries")


def main(image_dir="data/page_images", out_npy="data/doubao_page_emb.npy"):
    fns = [fn for fn in os.listdir(image_dir) if fn.lower().endswith(".png")]
    fns.sort(key=sort_key_png)
    paths = [os.path.join(image_dir, fn) for fn in fns]

    total = len(paths)
    print(f"Found {total} page images", flush=True)

    embeddings = []
    for i, path in enumerate(paths):
        print(f"[{i+1}/{total}] {os.path.basename(path)}", flush=True)
        b64 = encode_image(path)
        emb = embed_image(b64)
        embeddings.append(emb)
        time.sleep(0.5)

    arr = np.array(embeddings, dtype=np.float32)
    np.save(out_npy, arr)
    print(f"Saved {arr.shape} -> {out_npy}", flush=True)


if __name__ == "__main__":
    main()
