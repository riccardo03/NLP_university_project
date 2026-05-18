"""
    Corpus    : Hendrycks MATH dataset + curated math facts and formulas
    Embedder  : allenai/specter
    Vector DB : FAISS IndexFlatIP (cosine via normalised vectors)
"""

import re
import os
from typing import Optional

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Math RAG resources (corpus, embedder, and retrieval)
# ---------------------------------------------------------------------------
_maths_embedder = None
_maths_index = None
_maths_passages = None

MATHS_EMBED_MODEL = "allenai/specter"
MATHS_TOP_K = 5


# ---------------------------------------------------------------------------
# LaTeX cleanup (the MATH dataset is heavy in LaTeX, which confuses embedders)
# ---------------------------------------------------------------------------
def _clean_latex(text: str) -> str:
    """Convert common LaTeX into plain-text equivalents for better embedding."""
    if not text:
        return text
    # Fractions and roots
    text = re.sub(r'\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}', r'(\1)/(\2)', text)
    text = re.sub(r'\\sqrt\s*\{([^{}]*)\}', r'sqrt(\1)', text)
    # Operators
    text = re.sub(r'\\times\b', '*', text)
    text = re.sub(r'\\cdot\b', '*', text)
    text = re.sub(r'\\div\b', '/', text)
    text = re.sub(r'\\leq?\b', '<=', text)
    text = re.sub(r'\\geq?\b', '>=', text)
    text = re.sub(r'\\neq?\b', '!=', text)
    text = re.sub(r'\\pm\b', '+/-', text)
    # Greek letters and common symbols
    for greek in ['pi', 'theta', 'alpha', 'beta', 'gamma', 'delta', 'sigma',
                  'mu', 'lambda', 'phi', 'rho', 'tau', 'omega', 'epsilon']:
        text = re.sub(rf'\\{greek}\b', greek, text)
    text = re.sub(r'\\sum\b', 'sum', text)
    text = re.sub(r'\\int\b', 'integral', text)
    text = re.sub(r'\\prod\b', 'product', text)
    text = re.sub(r'\\infty\b', 'infinity', text)
    # Strip leftover LaTeX commands and grouping braces
    text = re.sub(r'\\[a-zA-Z]+\*?\s*', ' ', text)
    text = re.sub(r'[{}]', ' ', text)
    text = re.sub(r'\$+', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# Corpus setup
# ---------------------------------------------------------------------------
def setup_maths_rag(embed_model: str = MATHS_EMBED_MODEL) -> None:
    """Build corpus and FAISS index for the math RAG. Idempotent."""
    global _maths_embedder, _maths_index, _maths_passages
    if _maths_embedder is not None:
        return

    import faiss
    from sentence_transformers import SentenceTransformer

    print("[Maths RAG] Building corpus...")
    passages = []

    dataset_candidates = [
        ("AI-MO/NuminaMath-CoT", None),
        ("openai/gsm8k", "main"),
        ("HuggingFaceH4/MATH-500", None),
    ]
    loaded_dataset = False
    try:
        from datasets import load_dataset
        for name, config in dataset_candidates:
            try:
                if config is not None:
                    math_ds = load_dataset(name, config, split="train")
                else:
                    math_ds = load_dataset(name, split="train")

                count_before = len(passages)
                for item in math_ds:
                    problem = (item.get("problem") or item.get("question") or item.get("query") or "").strip()
                    solution = (item.get("solution") or item.get("answer") or item.get("response") or "").strip()

                    if problem and len(problem.split()) >= 3:
                        passages.append(_clean_latex(problem))

                    if solution and len(solution.split()) >= 5:
                        sol_excerpt = " ".join(solution.split()[:150])
                        cleaned = _clean_latex(sol_excerpt)
                        if cleaned and len(cleaned.split()) >= 5:
                            passages.append(cleaned)

                added = len(passages) - count_before
                print(f"      Loaded {name}: +{added:,} passages")
                loaded_dataset = True
                break
            except Exception as inner_e:
                print(f"      [Maths RAG] Could not load {name}: {type(inner_e).__name__}: {inner_e}")
                continue
    except Exception as e:
        print(f"  [Maths RAG] Warning: datasets library issue: {e}")

    if not loaded_dataset:
        print("      [Maths RAG] No external dataset loaded.")

    # Deduplicate while preserving order
    seen, unique = set(), []
    for p in passages:
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            unique.append(p)
    _maths_passages = unique

    if not _maths_passages:
        raise RuntimeError("Maths RAG corpus is empty after loading!")

    print(f"      {len(_maths_passages):,} passages total")

    print(f"[Maths RAG] Embedding with {embed_model} ...")
    _maths_embedder = SentenceTransformer(embed_model, device="cuda")
    emb = _maths_embedder.encode(
        _maths_passages,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
    embed_dim = emb.shape[1]
    print(f"[Maths RAG] Embedding dimension: {embed_dim}")
    _maths_index = faiss.IndexFlatIP(embed_dim)
    _maths_index.add(emb)


def maths_retrieve(query: str, k: int = MATHS_TOP_K) -> list:
    """Retrieve top-k passages for the math query."""
    global _maths_embedder, _maths_index, _maths_passages
    if _maths_embedder is None:
        raise RuntimeError("Maths RAG is not initialized. Call setup_maths_rag() first.")
    q = _maths_embedder.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")
    _, idx = _maths_index.search(q, k)
    return [_maths_passages[i] for i in idx[0]]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def rag_maths(query: str, option_texts: Optional[list] = None) -> str:

    if query is None:
        return ""
    stem = query.strip()
    if stem.upper().startswith("Q:"):
        stem = stem[2:].strip()

    # Build the retrieval query. Options are short for math (often just
    # numbers), so they add little signal; the stem dominates.
    if option_texts:
        options_str = " ".join(str(o).strip() for o in option_texts if o)
        retrieval_query = f"{stem} {options_str}".strip()
    else:
        retrieval_query = stem

    contexts = maths_retrieve(retrieval_query, k=MATHS_TOP_K)
    return "\n\n".join(contexts)