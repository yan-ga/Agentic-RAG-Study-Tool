"""
Pre-compute page embeddings using Qwen3-VL-Embedding model.
Saves embeddings to data/qwen3vl_page_emb.npy
"""

import os
import sys
import json
import numpy as np
import torch
from tqdm import tqdm

# Add the Qwen3-VL-Embedding source to path
sys.path.insert(0, "/Users/yan_ga/Desktop/Replicate Qwen-VL-Embedding/Qwen3-VL-Embedding-Mac/src")
from models.qwen3_vl_embedding import Qwen3VLEmbedder

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
CHUNKS_PATH = os.path.join(DATA_DIR, "new_chunks.jsonl")
PAGE_IMAGES_DIR = os.path.join(DATA_DIR, "page_images")
OUTPUT_PATH = os.path.join(DATA_DIR, "qwen3vl_page_emb.npy")

MODEL_PATH = "/Users/yan_ga/Desktop/Replicate Qwen-VL-Embedding/Qwen3-VL-Embedding-Mac/models/Qwen3-VL-Embedding-2B"

# Instructions for retrieval task
DOC_INSTRUCTION = "Represent this lecture slide for retrieval."


def load_chunks(chunks_jsonl):
    """Load chunks from JSONL file."""
    chunks = []
    with open(chunks_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def main():
    print("=" * 60)
    print("Computing Qwen3-VL Page Embeddings")
    print("=" * 60)

    # Load chunks to get image paths
    print("\nLoading chunks...")
    chunks = load_chunks(CHUNKS_PATH)
    print(f"  Loaded {len(chunks)} chunks")

    # Build image paths
    image_paths = []
    for c in chunks:
        doc_id = c["doc_id"]
        page_no = int(c["page_no"])
        img_path = os.path.join(PAGE_IMAGES_DIR, f"{doc_id}_p{page_no:03d}.png")
        image_paths.append(img_path)

    # Check all images exist
    missing = [p for p in image_paths if not os.path.exists(p)]
    if missing:
        print(f"  WARNING: {len(missing)} images not found")
        print(f"  First missing: {missing[0]}")

    # Load model
    print("\nLoading Qwen3-VL-Embedding model...")
    print(f"  Model path: {MODEL_PATH}")

    # Use reduced resolution for speed (~200K pixels instead of ~1.8M)
    IMAGE_FACTOR = 28
    embedder = Qwen3VLEmbedder(
        model_name_or_path=MODEL_PATH,
        torch_dtype=torch.float32,  # MPS works better with float32
        min_pixels=4 * IMAGE_FACTOR * IMAGE_FACTOR,
        max_pixels=256 * IMAGE_FACTOR * IMAGE_FACTOR,
    )
    print(f"  Model loaded on device: {embedder.model.device}")

    # Compute embeddings with batching
    batch_size = 8
    print(f"\nComputing embeddings for {len(image_paths)} pages (batch_size={batch_size})...")
    embeddings = []

    for start in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[start:start + batch_size]

        # Build batch inputs (only existing images)
        batch_inputs = []
        for img_path in batch_paths:
            if os.path.exists(img_path):
                batch_inputs.append({
                    "image": img_path,
                    "instruction": DOC_INSTRUCTION
                })
            else:
                batch_inputs.append(None)

        # Process valid inputs
        valid_inputs = [x for x in batch_inputs if x is not None]
        if valid_inputs:
            batch_embs = embedder.process(valid_inputs)
            batch_embs = batch_embs.cpu().numpy().astype(np.float32)

            # Assign embeddings back in order
            emb_idx = 0
            for inp in batch_inputs:
                if inp is not None:
                    embeddings.append(batch_embs[emb_idx:emb_idx + 1])
                    emb_idx += 1
                else:
                    embeddings.append(np.zeros((1, embedder.model.config.hidden_size), dtype=np.float32))
        else:
            for _ in batch_paths:
                embeddings.append(np.zeros((1, embedder.model.conelfig.hidden_size), dtype=np.float32))

    # Stack all embeddings
    embeddings = np.vstack(embeddings)
    print(f"\nEmbeddings shape: {embeddings.shape}")

    # Save
    print(f"\nSaving to {OUTPUT_PATH}...")
    np.save(OUTPUT_PATH, embeddings)
    print("Done!")


if __name__ == "__main__":
    main()
