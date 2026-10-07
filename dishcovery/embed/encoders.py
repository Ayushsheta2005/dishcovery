"""Encoders.

* ``SentenceTransformerEncoder`` — all-MiniLM-L6-v2 (384-d) by default; what
  the real experiments use (GPU recommended for 10M docs).
* ``HashingEncoder`` — dependency-free random projection of hashed word and
  character n-grams, L2-normalised. It captures surface overlap only (no
  semantics) and exists so the full pipeline runs on a laptop / in CI without
  torch. Never report semantic-retrieval results from it.

Embeddings are written as a raw little-endian float32 matrix (``.f32``), one
row per line of docs.jsonl, which the C++ indexer memory-maps.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Iterable, Iterator, List, Optional, Sequence

import numpy as np

from dishcovery.qu.text import normalize


def doc_text(doc: dict) -> str:
    """The text a document is embedded from (kept short: MiniLM truncates at 256 tokens)."""
    parts = [doc.get("name", "")]
    if doc.get("type") == "dish":
        parts.append(f"dish from {doc.get('restaurant_name', '')}")
    else:
        parts.append("restaurant")
    parts.append(" ".join(c.replace("_", " ") for c in doc.get("cuisines", [])))
    parts.append(" ".join(doc.get("tags", [])[:8]))
    parts.append(doc.get("description", "")[:300])
    return ". ".join(p for p in parts if p)


class Encoder:
    name = "base"
    dim = 0

    def encode(self, texts: Sequence[str], batch_size: int = 256) -> np.ndarray:
        raise NotImplementedError

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class HashingEncoder(Encoder):
    name = "hashing"

    def __init__(self, dim: int = 384, seed: int = 13) -> None:
        self.dim = dim
        self.seed = seed

    def _features(self, text: str) -> List[tuple]:
        toks = normalize(text).split()
        feats = [(t, 1.0) for t in toks]
        feats += [(f"{a}_{b}", 0.7) for a, b in zip(toks, toks[1:])]
        for t in toks:
            w = f"<{t}>"
            feats += [(w[i:i + 3], 0.3) for i in range(max(1, len(w) - 2))]
        return feats

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for f, w in self._features(text):
            h = int(hashlib.blake2b(f"{self.seed}|{f}".encode(), digest_size=8).hexdigest(), 16)
            v[h % self.dim] += w if (h >> 32) & 1 else -w
            v[(h >> 16) % self.dim] += 0.5 * (w if (h >> 48) & 1 else -w)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def encode(self, texts: Sequence[str], batch_size: int = 256) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts]).astype(np.float32) if texts else np.zeros((0, self.dim), np.float32)


class SentenceTransformerEncoder(Encoder):
    name = "sentence-transformers"

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None) -> None:
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.model_name = model
        self.model = SentenceTransformer(model, device=device)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str], batch_size: int = 256) -> np.ndarray:
        return self.model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


def load_encoder(spec: str = "hashing", dim: int = 384, device: Optional[str] = None) -> Encoder:
    """"hashing" or a sentence-transformers model name."""
    if spec in ("hashing", "hash"):
        return HashingEncoder(dim)
    return SentenceTransformerEncoder(spec, device=device)


def _batches(path: str, size: int) -> Iterator[List[str]]:
    batch: List[str] = []
    with open(path) as f:
        for line in f:
            batch.append(doc_text(json.loads(line)))
            if len(batch) == size:
                yield batch
                batch = []
    if batch:
        yield batch


def embed_docs(docs_path: str, out_path: str, encoder: Encoder, batch_size: int = 512,
               log_every_s: float = 30.0) -> dict:
    """Stream docs.jsonl -> float32 matrix file. Resumable: skips rows already written."""
    row_bytes = 4 * encoder.dim
    done = os.path.getsize(out_path) // row_bytes if os.path.exists(out_path) else 0
    t0 = last = time.time()
    n = 0
    with open(out_path, "ab") as out:
        if done:
            out.truncate(done * row_bytes)
        for batch in _batches(docs_path, batch_size):
            if n + len(batch) <= done:
                n += len(batch)
                continue
            skip = max(0, done - n)
            emb = encoder.encode(batch[skip:], batch_size=batch_size)
            out.write(np.ascontiguousarray(emb, dtype="<f4").tobytes())
            n += len(batch)
            if time.time() - last > log_every_s:
                last = time.time()
                print(f"embedded {n} docs ({n / (last - t0):.0f}/s)", flush=True)
    meta = {"encoder": encoder.name, "model": getattr(encoder, "model_name", encoder.name), "dim": encoder.dim,
            "rows": n, "seconds": round(time.time() - t0, 2)}
    with open(out_path + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta
