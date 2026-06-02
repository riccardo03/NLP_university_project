"""
Shared utilities for all RAG modules: HTML stripping, article fetch,
DDG search, stop-words base, and GLiNER singleton.
"""

import html as _html_module
import re

import requests

# ---------------------------------------------------------------------------
# Stop words
# ---------------------------------------------------------------------------

STOP_WORDS_BASE: frozenset[str] = frozenset({
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "and", "or", "as", "is", "are", "was", "were", "be", "been", "being",
    "what", "which", "who", "when", "where", "why", "how", "does", "do", "did",
    "has", "have", "had", "will", "would", "could", "should", "can", "may",
    "this", "that", "these", "those", "their", "there", "according", "following",
    "describes", "describe", "best", "most", "called", "named", "article",
})

# ---------------------------------------------------------------------------
# Skip lists (shared by entertainment + news)
# ---------------------------------------------------------------------------

SKIP_DOMAINS: frozenset[str] = frozenset({
    "youtube.com", "instagram.com", "tiktok.com",
    "twitter.com", "x.com", "linkedin.com",
    "reddit.com", "vk.com", "t.me",
})

SKIP_URL_PATTERNS: frozenset[str] = frozenset({
    "/tag/", "/tags/", "/category/", "/categories/",
    "/topic/", "/topics/", "/search/", "/archive/",
})

# ---------------------------------------------------------------------------
# Regex
# ---------------------------------------------------------------------------

TOKEN_RE = re.compile(r'[a-zA-ZÀ-ÿ0-9]+')

_BLOCK_TAG_RE = re.compile(
    r'<(script|style|nav|header|footer|aside|noscript)[^>]*>.*?</\1>',
    re.DOTALL | re.IGNORECASE,
)
_TAG_RE   = re.compile(r'<[^>]+>')
_WS_RE    = re.compile(r'\s+')
_P_TAG_RE = re.compile(r'<p[^>]*>(.*?)</p>', re.DOTALL | re.IGNORECASE)

# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------

def strip_html(raw: str) -> str:
    text = _BLOCK_TAG_RE.sub(" ", raw)
    text = _TAG_RE.sub(" ", text)
    text = _html_module.unescape(text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def extract_body(raw_html: str, max_chars: int = 7000) -> str:
    paragraphs = _P_TAG_RE.findall(raw_html)
    body_parts: list[str] = []
    total = 0

    for p in paragraphs:
        clean = _WS_RE.sub(" ", _TAG_RE.sub("", _html_module.unescape(p))).strip()
        if len(clean) >= 80:
            body_parts.append(clean)
            total += len(clean) + 1
            if total >= max_chars:
                break

    if body_parts:
        return " ".join(body_parts)[:max_chars]

    return strip_html(raw_html)[:max_chars]


# ---------------------------------------------------------------------------
# HTTP fetch
# ---------------------------------------------------------------------------

def fetch_article(url: str, *, user_agent: str = "RAGBot/1.0", timeout: int = 4) -> str:
    try:
        r = requests.get(
            url,
            timeout=(timeout, timeout),
            headers={"User-Agent": f"Mozilla/5.0 (compatible; {user_agent})"},
        )
        if r.status_code != 200:
            return ""
        return extract_body(r.text)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# DDG search + article fetch pipeline
# ---------------------------------------------------------------------------

def ddg_search(
    query: str,
    max_results: int = 3,
    *,
    timeout: int = 4,
    full: bool = False,
) -> list:
    """
    full=False → list[str] (URLs only).
    full=True  → list[dict] with keys: url, title, body.
    """
    try:
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results, timeout=timeout):
                url = r.get("href") or r.get("url", "")
                if url:
                    if full:
                        results.append({
                            "url": url,
                            "title": r.get("title", ""),
                            "body": r.get("body", ""),
                        })
                    else:
                        results.append(url)
        return results
    except Exception as e:
        print(f"  [RAG] DDG error: {e}")
        return []


def search_and_fetch(
    query: str,
    q_keywords: set[str],
    *,
    max_results: int = 3,
    target: int | None = None,
    min_chars: int = 300,
    timeout: int = 4,
    user_agent: str = "RAGBot/1.0",
    module_tag: str = "RAG",
) -> list[tuple[str, str]]:
    """DDG search → fetch → keyword filter. target=None means no limit."""
    results = []
    for url in ddg_search(query, max_results=max_results, timeout=timeout):
        if target is not None and len(results) >= target:
            break
        if any(domain in url for domain in SKIP_DOMAINS):
            print(f"  [{module_tag}] skipping: {url[:80]}")
            continue
        if any(pat in url for pat in SKIP_URL_PATTERNS):
            print(f"  [{module_tag}] skipping: {url[:80]}")
            continue
        text = fetch_article(url, user_agent=user_agent, timeout=timeout)
        if text and len(text) >= min_chars and any(kw in text.lower() for kw in q_keywords):
            results.append((text, url))
    return results


# ---------------------------------------------------------------------------
# GLiNER singleton
# ---------------------------------------------------------------------------

GLINER_MODEL_NAME   = "urchade/gliner_medium-v2.1"
_gliner_model       = None
_gliner_model_tried = False


def get_gliner_model():
    global _gliner_model, _gliner_model_tried
    if _gliner_model_tried:
        return _gliner_model
    _gliner_model_tried = True
    try:
        from gliner import GLiNER
        import torch
        _gliner_model = GLiNER.from_pretrained(GLINER_MODEL_NAME)
        _gliner_model.eval()
        torch.set_grad_enabled(False)
    except Exception as e:
        print(f"  [RAG] GLiNER model unavailable: {e}")
    return _gliner_model


def setup_gliner() -> None:
    """Eagerly load GLiNER. Idempotent."""
    if _gliner_model_tried:
        return
    print(f"[RAG] Loading GLiNER model ({GLINER_MODEL_NAME})...")
    get_gliner_model()
    if _gliner_model is not None:
        print("[RAG] GLiNER ready.")
    else:
        print("[RAG] GLiNER unavailable; will fall back to regex extraction.")


# ---------------------------------------------------------------------------
# Subject extraction
# ---------------------------------------------------------------------------

_QUOTED_RE        = re.compile(r"""['\"''""]([\w][\w\s,.\-&!]{1,58}?)['\"''""]""")
_PROPER_MULTI_RE  = re.compile(r'\b[A-ZÀ-Ý][a-zA-ZÀ-ÿ]+(?:\s+[A-ZÀ-Ý][a-zA-ZÀ-ÿ]+)+\b')
_PROPER_SINGLE_RE = re.compile(r'^[A-ZÀ-Ý][a-zA-ZÀ-ÿ]{2,}$')


def extract_subjects_gliner(
    text: str,
    labels: list[str],
    label_priority: dict[str, int] | None = None,
    threshold: float = 0.5,
) -> list[tuple[str, str]]:
    """Returns [(text, label), ...] sorted by label priority, deduped."""
    model = get_gliner_model()
    if model is None:
        return []
    try:
        import torch
        with torch.no_grad():
            entities = model.predict_entities(text, labels, threshold=threshold)
        if label_priority:
            entities.sort(key=lambda e: (label_priority.get(e["label"], 99), e["start"]))
        seen: set[str] = set()
        result: list[tuple[str, str]] = []
        for e in entities:
            t = e["text"].strip()
            tl = t.lower()
            if t and tl not in seen:
                seen.add(tl)
                result.append((t, e["label"]))
        return result
    except Exception:
        return []


def extract_subjects_regex(
    question: str,
    stop_words: frozenset[str] = STOP_WORDS_BASE,
) -> list[str]:
    """Quoted titles first, then multi-word proper nouns, then single proper nouns."""
    subjects: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        s = raw.strip()
        sl = s.lower()
        if s and sl not in seen and sl not in stop_words:
            seen.add(sl)
            subjects.append(s)

    for q in _QUOTED_RE.findall(question):
        _add(q)
    for m in _PROPER_MULTI_RE.findall(question):
        _add(m)
    for w in question.split()[1:]:
        clean = re.sub(r"[^\w]+$", "", w)
        if (_PROPER_SINGLE_RE.match(clean)
                and clean.lower() not in seen
                and not any(clean in s.split() for s in subjects)):
            _add(clean)

    return subjects
