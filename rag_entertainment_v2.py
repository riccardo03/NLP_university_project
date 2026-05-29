import re
import urllib.parse
import requests
import concurrent.futures

from gliner import GLiNER

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

_WIKI_UA = "QuizBot/1.0"
_TIMEOUT = 4

model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

LABELS = [    "movie", "film", "TV show", "TV series", "person", "actor", "musician",
    "director", "band", "music group", "album", "song", "family member",
    "historical person", "character"]

# ─────────────────────────────────────────────
# STOPWORDS (pulite)
# ─────────────────────────────────────────────

_STOP_WORDS = {
    "the","a","an","of","in","on","at","to","for","with","by","from",
    "and","or","as","is","are","was","were","be","been","being",
    "what","which","who","when","where","why","how","does","do","did",
    "has","have","had","will","would","could","should","can","may",
    "this","that","these","those"
}

_RELATION_VERBS = {
    "starred","played","appeared","performed","voiced","portrayed",
    "directed","wrote","produced","created","released","featured",
    "sang","won","nominated","known","called","named","is","was"
}

_TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ0-9$!&]+")

# ─────────────────────────────────────────────
# TOKEN + KEYWORDS
# ─────────────────────────────────────────────

def _tokenize(text):
    return _TOKEN_RE.findall(text.lower())

def _keywords(text):
    return {t for t in _tokenize(text) if len(t) >= 3 and t not in _STOP_WORDS}

# ─────────────────────────────────────────────
# GLINER SUBJECT EXTRACTION (CORE FIX)
# ─────────────────────────────────────────────

def _extract_subjects(text: str):
    ents = model.predict_entities(text, LABELS)

    subjects = []
    seen = set()

    for e in ents:
        ent = e["text"].strip()
        k = ent.lower()

        if k not in seen and k not in _STOP_WORDS:
            seen.add(k)
            subjects.append(ent)

    return subjects

# ─────────────────────────────────────────────
# WIKIPEDIA (robusto)
# ─────────────────────────────────────────────

def _wiki_lookup(query: str):
    try:
        url = (
            "https://en.wikipedia.org/w/api.php"
            f"?action=query&list=search&srsearch={urllib.parse.quote(query)}&format=json"
        )

        r = requests.get(url, headers={"User-Agent": _WIKI_UA}, timeout=_TIMEOUT)
        results = r.json().get("query", {}).get("search", [])

        if not results:
            return ""

        title = results[0]["title"]

        url2 = (
            "https://en.wikipedia.org/w/api.php"
            f"?action=query&prop=extracts&explaintext=true&titles={urllib.parse.quote(title)}&format=json"
        )

        r2 = requests.get(url2, headers={"User-Agent": _WIKI_UA}, timeout=_TIMEOUT)
        pages = r2.json()["query"]["pages"]

        text = next(iter(pages.values())).get("extract", "")

        if "may refer to" in text.lower():
            return ""

        return text

    except Exception:
        return ""

def _wiki_passages(text: str, query: str, max_chars=1200):
    if not text:
        return ""

    paragraphs = [
        p.strip() for p in text.split("\n")
        if len(p.strip()) > 60
    ]

    if not paragraphs:
        return text[:max_chars]

    qk = _keywords(query)

    intro = paragraphs[0]
    rest = paragraphs[1:]

    scored = []
    for p in rest:
        toks = set(_tokenize(p))
        scored.append((len(qk & toks), p))

    scored.sort(reverse=True)

    out = [intro]
    budget = max_chars - len(intro)

    for score, p in scored:
        if budget <= 100 or score < 2:
            break

        chunk = p[:budget]
        out.append(chunk)
        budget -= len(chunk)

    return "\n\n".join(out)

# ─────────────────────────────────────────────
# DDG
# ─────────────────────────────────────────────

def _ddg(query: str, k=2):
    try:
        from ddgs import DDGS
        with DDGS() as d:
            return [r["body"] for r in d.text(query, max_results=k) if r.get("body")]
    except Exception:
        return []

# ─────────────────────────────────────────────
# SCORING (FIXED & STABLE)
# ─────────────────────────────────────────────

def _score(option, subjects, snippets):
    opt_kws = _keywords(option)
    if not opt_kws:
        return 0.0

    subj_kws = set()
    for s in subjects:
        subj_kws |= _keywords(s)

    best = 0.0

    for snip in snippets:
        toks = set(_tokenize(snip))

        overlap = len(opt_kws & toks) / (len(opt_kws) + 1e-6)

        subj_bonus = 1.0 if subj_kws & toks else 0.0

        relation = sum(1 for t in toks if t in _RELATION_VERBS)
        relation = min(relation * 0.1, 0.5)

        score = overlap + 0.3 * subj_bonus + relation

        best = max(best, score)

    return best

# ─────────────────────────────────────────────
# PIPELINE PRINCIPALE
# ─────────────────────────────────────────────

def rag_entertainment(query: str, option_texts=None, num_results=3):
    print(f"[rag_entertainment] query: {query!r}")

    subjects = _extract_subjects(query)
    print(f"[rag_entertainment] subjects: {subjects}")

    main = subjects[0] if subjects else query
    subj_str = " ".join(subjects[:2]) if subjects else query

    wiki = _wiki_lookup(main)
    wiki_ctx = _wiki_passages(wiki, query)
    print(f"[rag_entertainment] wiki for {main!r}: {'found' if wiki_ctx else 'not found'} ({len(wiki_ctx)} chars)")

    ddg_global = _ddg(subj_str, num_results)
    print(f"[rag_entertainment] ddg global: {len(ddg_global)} result(s)")

    shared = ([wiki_ctx] if wiki_ctx else []) + ddg_global

    # ── NO OPTIONS MODE ──
    if not option_texts:
        return "\n\n".join(shared)[:1500]

    # ── PARALLEL OPTION SEARCH ──
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(option_texts)) as ex:
        futures = [
            ex.submit(_ddg, f"{subj_str} {opt}", 2)
            for opt in option_texts
        ]
        opt_snips = [f.result() for f in futures]
    print(f"[rag_entertainment] per-option ddg fetched for {len(option_texts)} options")

    # ── SCORING ──
    scores = {}
    for i, opt in enumerate(option_texts):
        scores[i] = _score(opt, subjects, opt_snips[i] + shared)

    ranked = sorted(scores, key=lambda i: scores[i], reverse=True)
    print(f"[rag_entertainment] scores: { {option_texts[i]: round(scores[i], 2) for i in ranked} }")

    # ── OUTPUT ──
    out = []

    if wiki_ctx:
        out.append("WIKIPEDIA:\n" + wiki_ctx)

    for i in ranked:
        out.append(
            f"[{i}] {option_texts[i]} | score={scores[i]:.2f}\n"
            f"{opt_snips[i][0][:300] if opt_snips[i] else '(no evidence)'}"
        )

    return "\n\n".join(out)[:2500]