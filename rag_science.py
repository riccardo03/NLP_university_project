"""
Science RAG pipeline.

Corpus    : SciQ + OpenBookQA + MMLU science subsets
Embedder  : BAAI/bge-small-en-v1.5
Vector DB : FAISS IndexFlatIP (cosine via normalised vectors)
Retrieval : Multi-query FAISS → BM25 hybrid rerank (RRF)
"""

import os
import re
from typing import Optional

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

_science_embedder = None
_science_index    = None
_science_passages = None
_bm25_index       = None

SCIENCE_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
SCIENCE_TOP_K       = 5
RRF_K               = 60

MMLU_SCIENCE_SUBSETS = [
    "high_school_biology", "high_school_chemistry", "high_school_physics",
    "college_biology",     "college_chemistry",     "college_physics",
    "astronomy", "anatomy", "nutrition", "virology", "medical_genetics",
    "environmental_science", "conceptual_physics",
]


def setup_science_rag(embed_model: str = SCIENCE_EMBED_MODEL) -> None:
    """Build corpus, BM25 index, and FAISS index. Idempotent."""
    global _science_embedder, _science_index, _science_passages, _bm25_index
    if _science_embedder is not None:
        return

    import faiss
    from datasets import load_dataset
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

    print("[Science RAG] Building corpus...")
    passages = []

    for item in load_dataset("allenai/sciq", split="train"):
        s = (item.get("support") or "").strip()
        if s and len(s.split()) >= 5:
            passages.append(s)

    for split in ("train", "validation", "test"):
        for item in load_dataset("allenai/openbookqa", "additional", split=split):
            f = (item.get("fact1") or "").strip()
            if f:
                passages.append(f)

    for subject in MMLU_SCIENCE_SUBSETS:
        try:
            for item in load_dataset("cais/mmlu", subject, split="test"):
                passages.append(f"{item['question']} {item['choices'][item['answer']]}.")
        except Exception as e:
            print(f"  [Science RAG] Skipping mmlu/{subject}: {e}")

    seen, unique = set(), []
    for p in passages:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    _science_passages = unique
    print(f"  {len(_science_passages):,} passages")

    _bm25_index = BM25Okapi([p.lower().split() for p in _science_passages])

    print(f"[Science RAG] Embedding with {embed_model} ...")
    _science_embedder = SentenceTransformer(embed_model, device="cuda")
    emb = _science_embedder.encode(
        _science_passages,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
    _science_index = faiss.IndexFlatIP(emb.shape[1])
    _science_index.add(emb)


def science_retrieve(query: str, k: int = SCIENCE_TOP_K) -> list:
    if _science_embedder is None:
        raise RuntimeError("Science RAG is not initialised. Call setup_science_rag() first.")
    q = _science_embedder.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")
    _, idx = _science_index.search(q, k)
    return [_science_passages[i] for i in idx[0]]


_QUESTION_RE = re.compile(
    r"^(?:Q:\s*)?(.*?)\s*\[0\]\s*(.*?)\s*\[1\]\s*(.*?)\s*\[2\]\s*(.*?)\s*\[3\]\s*(.*?)\s*$",
    re.DOTALL,
)


def _parse_question(text: str):
    m = _QUESTION_RE.match(text.strip())
    if not m:
        return None, None
    stem = m.group(1).strip().rstrip(".")
    options = [m.group(i).strip().rstrip(".") for i in range(2, 6)]
    return stem, options


def multi_query_retrieve(stem: str, options: list, k_stem: int = 3, k_opt: int = 1) -> list:
    seen, results = set(), []
    for p in science_retrieve(stem, k=k_stem):
        if p not in seen:
            seen.add(p)
            results.append(p)
    for opt in options:
        for p in science_retrieve(f"{stem} {opt}", k=k_opt):
            if p not in seen:
                seen.add(p)
                results.append(p)
    return results


def hybrid_retrieve(query: str, passages: list, k: int = 5, alpha: float = 0.6) -> list:
    if not passages:
        return passages

    import faiss

    embs = _science_embedder.encode(
        passages, normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")
    tmp_index = faiss.IndexFlatIP(embs.shape[1])
    tmp_index.add(embs)
    q_emb = _science_embedder.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")
    _, dense_idx = tmp_index.search(q_emb, len(passages))
    dense_ranks = {passages[i]: rank for rank, i in enumerate(dense_idx[0])}

    all_bm25 = _bm25_index.get_scores(query.lower().split())
    p_to_idx = {p: i for i, p in enumerate(_science_passages)}
    sparse_ranks = {
        p: rank
        for rank, p in enumerate(
            sorted(passages, key=lambda p: all_bm25[p_to_idx[p]], reverse=True)
        )
    }

    def rrf(rank: int) -> float:
        return 1.0 / (RRF_K + rank + 1)

    scores = {
        p: alpha * rrf(dense_ranks[p]) + (1 - alpha) * rrf(sparse_ranks[p])
        for p in passages
    }
    return sorted(passages, key=lambda p: scores[p], reverse=True)[:k]


def rag_science(query: str, option_texts: Optional[list] = None) -> str:
    if option_texts is not None:
        if len(option_texts) != 4:
            raise ValueError(f"Expected 4 options, got {len(option_texts)}.")
        stem = query.strip()
        if stem.upper().startswith("Q:"):
            stem = stem[2:].strip()
        options = [str(o).strip().rstrip(".") for o in option_texts]
    else:
        stem, options = _parse_question(query)
        if stem is None:
            raise ValueError(
                "Could not parse [0]/[1]/[2]/[3] options from query. "
                "Pass option_texts explicitly."
            )

    candidates = multi_query_retrieve(stem, options)
    return "\n\n".join(hybrid_retrieve(stem, candidates, k=5, alpha=0.6))
