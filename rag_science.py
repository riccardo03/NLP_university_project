"""
    Corpus    : SciQ supporting paragraphs + OpenBookQA core facts (~18k passages)
    Embedder  : BAAI/bge-small-en-v1.5         (~500 MB VRAM)
    Vector DB : FAISS IndexFlatIP (cosine via normalised vectors)
    Generator : Qwen/Qwen2.5-7B-Instruct, 4-bit NF4 via bitsandbytes (~6 GB VRAM)
"""

import re
from typing import Optional
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Science RAG resources (corpus, embedder, and retrieval)
# ---------------------------------------------------------------------------
_science_embedder = None
_science_index = None
_science_passages = None

SCIENCE_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
SCIENCE_TOP_K = 5


def setup_science_rag(embed_model: str = SCIENCE_EMBED_MODEL) -> None:
    """Build corpus and FAISS index for the science RAG. Idempotent."""
    global _science_embedder, _science_index, _science_passages
    if _science_embedder is not None:
        return

    import faiss
    from sentence_transformers import SentenceTransformer
    from datasets import load_dataset

    print("[Science RAG] Building corpus...")
    passages = []

    sciq = load_dataset("allenai/sciq", split="train")
    for item in sciq:
        s = (item.get("support") or "").strip()
        if s and len(s.split()) >= 5:
            passages.append(s)

    for split in ("train", "validation", "test"):
        obqa = load_dataset("allenai/openbookqa", "additional", split=split)
        for item in obqa:
            f = (item.get("fact1") or "").strip()
            if f:
                passages.append(f)

    seen, unique = set(), []
    for p in passages:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    _science_passages = unique
    print(f"      {_science_passages and len(_science_passages) or 0:,} passages")

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
    """Retrieve top-k passages for the science query.

    This function assumes that setup_science_rag() was run externally,
    for example from millionaire_bot.load_model().
    """
    global _science_embedder, _science_index, _science_passages
    if _science_embedder is None:
        raise RuntimeError("Science RAG is not initialized. Call setup_science_rag() first.")
    print(f"  [RAG-Sci] FAISS search | query: {query[:80]!r} | k={k}")
    q = _science_embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    distances, idx = _science_index.search(q, k)
    results = [_science_passages[i] for i in idx[0]]
    print(f"  [RAG-Sci] Top-{k} scores: {[round(float(d), 3) for d in distances[0]]}")
    for rank, (passage, score) in enumerate(zip(results, distances[0])):
        print(f"    [{rank}] score={score:.3f} | {passage[:100]}…")
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_QUESTION_RE = re.compile(
    r"^(?:Q:\s*)?(.*?)\s*\[0\]\s*(.*?)\s*\[1\]\s*(.*?)"
    r"\s*\[2\]\s*(.*?)\s*\[3\]\s*(.*?)\s*$",
    re.DOTALL,
)

def _parse_question(text: str):
    m = _QUESTION_RE.match(text.strip())
    if not m:
        return None, None
    stem = m.group(1).strip().rstrip(".").strip()
    options = [m.group(i).strip().rstrip(".") for i in range(2, 6)]
    return stem, options


def _retrieve(query: str, k: int = None):
    return science_retrieve(query, k=k if k is not None else SCIENCE_TOP_K)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def rag_science(query: str, option_texts: Optional[list] = None) -> str:
    # RAG setup is performed externally (e.g. by millionaire_bot.load_model()).
    # This keeps rag_science fast during import and allows setup to run
    # only when the environment and model are ready.

    # ---- Resolve stem and options ---------------------------------------
    if option_texts is not None:
        if len(option_texts) != 4:
            raise ValueError(f"Expected 4 options, got {len(option_texts)}.")
        stem = query.strip()
        if stem.upper().startswith("Q:"):
            stem = stem[2:].strip()
        options = [str(o).strip().rstrip(".") for o in option_texts]
        print(f"  [RAG-Sci] stem (from arg): {stem!r}")
    else:
        stem, options = _parse_question(query)
        if stem is None:
            raise ValueError(
                "Could not parse [0]/[1]/[2]/[3] options from query, "
                "and no option_texts argument was provided."
            )
        print(f"  [RAG-Sci] stem (parsed): {stem!r}")

    print(f"  [RAG-Sci] options: {options}")

    # ---- Retrieve (delegated) -------------------------------------------
    # Including options in the query covers the answer space, not just the stem.
    retrieval_query = stem + " " + " ".join(options)
    print(f"  [RAG-Sci] retrieval query: {retrieval_query[:120]!r}")
    contexts = _retrieve(retrieval_query, k=None)
    print(f"  [RAG-Sci] retrieved {len(contexts)} passages, total chars: {sum(len(c) for c in contexts)}")

    return "\n\n".join(contexts)


